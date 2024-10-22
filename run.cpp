#include "run.h"

//============== UTILS FUNCTIONS ===============
void parse_args(int argc, char *argv[], std::string& model_weights_path, std::string& vocab_json_path,
                std::string& prompt, int& n_tokens_to_predict, int& seed) {
    if (argc < 3) {
        std::cerr << "model weights or/and vocab are not provided" << std::endl;
        return exit(1);
    }

    model_weights_path = argv[1];
    vocab_json_path = argv[2];
    for (int i = 3; i < argc; i += 2) {
        if (std::string(argv[i]) == "-p") prompt = argv[i + 1];
        else if (std::string(argv[i]) == "-n")
            n_tokens_to_predict = atoi(argv[i + 1]);
        else if (std::string(argv[i]) == "-s")
            seed = atoi(argv[i + 1]);
    }
}

WeightPointers set_weight_pointers(float* all_weights, size_t total_params) {
    WeightPointers ptrs{};
    size_t offset = 0;

    // Embedding weights
    ptrs.wte_w = all_weights + offset;
    offset += Config::vocab_size * Config::dim;

    ptrs.wpe_w = all_weights + offset;
    offset += Config::seq_len * Config::dim;

    // Transformer blocks
    for (int i = 0; i < Config::n_layers; ++i) {
        // LayerNorm1 weights and biases
        ptrs.ln1_w[i] = all_weights + offset;
        offset += Config::dim;
        ptrs.ln1_b[i] = all_weights + offset;
        offset += Config::dim;

        // Self-Attention c_attn weights and biases
        ptrs.attn_c_attn_w[i] = all_weights + offset;
        offset += Config::dim * (Config::dim * 3);
        ptrs.attn_c_attn_b[i] = all_weights + offset;
        offset += Config::dim * 3;

        // Self-Attention c_proj weights and biases
        ptrs.attn_c_proj_w[i] = all_weights + offset;
        offset += Config::dim * Config::dim;
        ptrs.attn_c_proj_b[i] = all_weights + offset;
        offset += Config::dim;

        // LayerNorm2 weights and biases
        ptrs.ln2_w[i] = all_weights + offset;
        offset += Config::dim;
        ptrs.ln2_b[i] = all_weights + offset;
        offset += Config::dim;

        // MLP c_fc weights and biases
        ptrs.mlp_c_fc_w[i] = all_weights + offset;
        offset += Config::dim * Config::hidden_dim;
        ptrs.mlp_c_fc_b[i] = all_weights + offset;
        offset += Config::hidden_dim;

        // MLP c_proj weights and biases
        ptrs.mlp_c_proj_w[i] = all_weights + offset;
        offset += Config::hidden_dim * Config::dim;
        ptrs.mlp_c_proj_b[i] = all_weights + offset;
        offset += Config::dim;
    }

    // Final LayerNorm weights and biases
    ptrs.ln_f_w = all_weights + offset;
    offset += Config::dim;
    ptrs.ln_f_b = all_weights + offset;
    offset += Config::dim;

    // LM Head weights and biases
    ptrs.lm_head_w = all_weights + offset;
    offset += Config::vocab_size * Config::dim;

    //Verify that the offset matches the total number of parameters
    if (offset != total_params) {
        std::cerr << "Error: Mismatch between expected and actual number of parameters." << std::endl;
    }

    return ptrs;
}


// Function to allocate memory for run_state variables
void allocate_run_state() {
    // Main buffers
    run_state.x = new float[Config::dim];
    run_state.xb = new float[Config::dim];

    // Embedding buffers
    run_state.emb_token_out = new float[Config::dim];
    run_state.emb_pos_out = new float[Config::dim];

    // Layer norm buffer
    run_state.ln_output = new float[Config::dim];

    // MLP buffer
    run_state.mlp_buffer = new float[Config::hidden_dim];

    // Transformer buffer
    run_state.attn_output = new float[Config::dim];

    // Self-Attention buffers
    run_state.qkv = new float[Config::dim * 3];
    run_state.attn = new float[Config::n_heads * Config::seq_len];
    run_state.y = new float[Config::dim];

    // Logits buffer
    run_state.logits = new float[Config::vocab_size];

    // KV cache
    run_state.key_cache = new float[Config::n_layers * Config::seq_len * Config::dim];
    run_state.value_cache = new float[Config::n_layers * Config::seq_len * Config::dim];
}

// Function to deallocate memory for run_state variables
void deallocate_run_state() {
    // Main buffers
    delete[] run_state.x;
    delete[] run_state.xb;

    // Embedding buffers
    delete[] run_state.emb_token_out;
    delete[] run_state.emb_pos_out;

    // Layer norm buffer
    delete[] run_state.ln_output;

    // MLP buffer
    delete[] run_state.mlp_buffer;

    // Transformer buffer
    delete[] run_state.attn_output;

    // Self-Attention buffers
    delete[] run_state.qkv;
    delete[] run_state.attn;
    delete[] run_state.y;

    // Logits buffer
    delete[] run_state.logits;

    // KV cache
    delete[] run_state.key_cache;
    delete[] run_state.value_cache;
}


// GELU function with tanh approximation
void gelu(float* x, int n_channels) {
    const float sqrt_2_over_pi = std::sqrt(2.0f / (float)M_PI); // sqrt(2/pi)
    const float coeff = 0.044715f;

    for (int i = 0; i < n_channels; ++i) {
        float x_i = x[i];
        float x_cubed = x_i * x_i * x_i;
        float tanh_term = std::tanh(sqrt_2_over_pi * (x_i + coeff * x_cubed));
        x[i] = 0.5f * x_i * (1.0f + tanh_term);
    }
}


void softmax(float* x, int n_channels) {
    float max_score = std::numeric_limits<float>::lowest();
    for (int t = 0; t < n_channels; t++) {
        if (x[t] > max_score) {
            max_score = x[t];
        }
    }

    float sum_exp = 0.0f;
    for (int t = 0; t < n_channels; t++) {
        x[t] = std::expf(x[t] - max_score);
        sum_exp += x[t];
    }

    for (int t = 0; t < n_channels; t++) {
        x[t] /= sum_exp;
    }
}


void Sampler::_topk(float *input, int input_size, int k, std::vector<float> &top_values, std::vector<int> &top_indices) {
    // Create a vector of pairs (value, index)
    std::vector<std::pair<float, int>> value_index_pairs;

    // Fill the vector with the values and their corresponding indices
    value_index_pairs.reserve(input_size);
    for (int i = 0; i < input_size; ++i) {
        value_index_pairs.emplace_back(input[i], i);
    }

    // Use partial_sort to get the k largest elements (sorted in descending order)
    std::partial_sort(
            value_index_pairs.begin(),
            value_index_pairs.begin() + k,
            value_index_pairs.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first; // Compare based on the values (descending order)
            }
    );

    // Extract the top-k values and their indices
    top_values.clear();
    top_indices.clear();
    for (int i = 0; i < k; ++i) {
        top_values.push_back(value_index_pairs[i].first);
        top_indices.push_back(value_index_pairs[i].second);
    }
}

size_t Sampler::_multinomial_sample(const std::vector<float> &probabilities) {
    // Generate a random number in the range [0, 1)
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double random_value = dis(_rd);

    // Compute the cumulative sum (CDF) of the probabilities
    double cumulative_sum = 0.0;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        cumulative_sum += probabilities[i];
        if (random_value < cumulative_sum) {
            return i;
        }
    }

    // In case of floating-point precision issues, return the last index
    return probabilities.size() - 1;
}

int Sampler::sample(float* logits) {
    softmax(logits, Config::vocab_size);
    //crop the probs to only the top k options to avoid choosing very unlikely tokens
    std::vector<float> top_values;
    std::vector<int> top_indices;
    _topk(logits, Config::vocab_size, Config::topk_val, top_values, top_indices);
    // sample from the distribution
    size_t idx = _multinomial_sample(top_values);

    return top_indices[idx];
}

//============== LAYERS FUNCTIONS (forward) ================
void Embedding::forward(const int token, float *output) {
    memcpy(output, _w + token*_embedding_dim, _embedding_dim*sizeof(*output));
}


void Linear::forward(const float* input, float* output) {
    for (int o = 0; o < _n_output_channels; o++) {
        output[o] = (_b != nullptr) ? _b[o] : 0.0f;
    }
    for (int i = 0; i < _n_input_channels; i++) {
        for (int o = 0; o < _n_output_channels; o++) {
            output[o] += input[i] * _w[i * _n_output_channels + o];
        }
    }
}


void LayerNorm::forward(const float* input, float* output) {
    // the C-dimensional vector of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    // calculate the mean
    float m = 0.0f;
    for (int i = 0; i < _n_channels; i++) {
        m += input[i];
    }
    m = m/(float)_n_channels;
    // calculate the variance (without any bias correction)
    float v = 0.0f;
    for (int i = 0; i < _n_channels; i++) {
        float xshift = input[i] - m;
        v += xshift * xshift;
    }
    v = v/(float)_n_channels;
    // calculate the rstd (reciprocal standard deviation)
    float s = 1.0f / sqrtf(v + eps);
    for (int i = 0; i < _n_channels; i++) {
        float n = (s * (input[i] - m)); // normalize
        float o = n * _w[i] + _b[i]; // scale and shift
        output[i] = o; // write
    }
}

void MLP::forward(const float *input, float *output) {
    _c_fc.forward(input, run_state.mlp_buffer);
    gelu(run_state.mlp_buffer, Config::hidden_dim);
    _c_proj.forward(run_state.mlp_buffer, output);
}

void SelfAttention::forward(const float *input, float *output) {
    _c_attn.forward(input, run_state.qkv);
    //storeFloatsToFile(run_state.qkv, Config::dim*3, "h0_check.txt");
    float *q = run_state.qkv; // (n_heads, head_size)
    float *k = run_state.qkv + Config::dim; // (n_heads, head_size)
    float *v = run_state.qkv + Config::dim*2; // (n_heads, head_size)
    //Cache K and V
    memcpy(_k_cache + _prev_length*Config::dim, k, Config::dim*sizeof(*k));
    memcpy(_v_cache + _prev_length*Config::dim, v, Config::dim*sizeof(*v));
    _prev_length++;

    float scale = 1.0f / std::sqrtf(Config::head_size);
    for (int t = 0; t < _prev_length; t++) {
        for (int h = 0; h < Config::n_heads; h++) {
            float dot_product = 0.0f;
            for (int i = 0; i < Config::head_size; i++) {
                // Compute the index for the query vector
                int q_index = h * Config::head_size + i;
                // Compute the index for the key vector at time t
                int k_index = t * Config::dim + h * Config::head_size + i;
                // Accumulate the dot product
                dot_product += q[q_index] * _k_cache[k_index];
            }
            // Store the attention score for head h at time t
            run_state.attn[h * _prev_length + t] = dot_product*scale;
        }
    }

    for (int h = 0; h < Config::n_heads; h++) {
        float *softmax_input = run_state.attn + h * _prev_length;
        softmax(softmax_input, _prev_length);
    }

    // Compute att * v
    for (int h = 0; h < Config::n_heads; h++) {
        for (int i = 0; i < Config::head_size; i++) {
            float sum = 0.0f;
            for (int t = 0; t < _prev_length; t++) {
                // Index for attention weight
                int att_idx = h * _prev_length + t;

                // Index for value vector
                int v_idx = t * Config::dim + h * Config::head_size + i;

                // Accumulate the weighted value
                sum += run_state.attn[att_idx] * _v_cache[v_idx];
            }
            // Store the result in the output buffer
            int out_idx = h * Config::head_size + i;
            run_state.y[out_idx] = sum;
        }
    }

    _c_proj.forward(run_state.y, output);
}

void TransformerBlock::forward(const float *input, float *output) {
    _ln1.forward(input, run_state.ln_output);
    _attn.forward(run_state.ln_output, run_state.attn_output);
    for (int i = 0; i < Config::dim; i++){
        run_state.attn_output[i] += input[i];
    }
    _ln2.forward(run_state.attn_output, run_state.ln_output);
    _mlp.forward(run_state.ln_output, output);
    for (int i = 0; i < Config::dim; i++){
        output[i] += run_state.attn_output[i];
    }
}

void GPT2Pretrained::forward(const int token, float *logits) {
    int pos = _prev_length++;
    _wte.forward(token, run_state.emb_token_out);
    _wpe.forward(pos, run_state.emb_pos_out);
    for (int i = 0; i < Config::dim; i++){
        run_state.x[i] = run_state.emb_token_out[i] + run_state.emb_pos_out[i];
    }

    //double buffering mechanism to reuse memory and avoid copying
    float* x1 = run_state.x;
    float* x2 = run_state.xb;
    float* tmp_ptr = nullptr;
    for (int i = 0; i < Config::n_layers; i++){
        _h[i].forward(x1, x2);
        tmp_ptr = x1;
        x1 = x2;
        x2 = tmp_ptr;
    }
    _ln_f.forward(x1, x2);
    _lm_head.forward(x2, logits);
}

//============== MAIN ================
int main(int argc, char *argv[]) {

    std::string model_weights_path;
    std::string vocab_json_path;
    std::string prompt;
    int n_tokens_to_predict = 200;
    int seed = -1;
    parse_args(argc, argv, model_weights_path, vocab_json_path, prompt, n_tokens_to_predict, seed);

    if (prompt.empty())
        std::cerr << "You do not specify prompt!\n";


    //Read all weights
    size_t total_params = 0;
    float* all_weights = read_checkpoint(model_weights_path, total_params);
    if (!all_weights) {
        return -1;
    }

    // Set up weight pointers
    WeightPointers ptrs = set_weight_pointers(all_weights, total_params);

    // Allocate memory for run_state variables
    allocate_run_state();

    //Set up KV cache pointers
    float* key_cache_ptrs[Config::n_layers];
    float* value_cache_ptrs[Config::n_layers];
    size_t cache_size_per_layer = Config::seq_len * Config::n_heads * Config::head_size;
    for (int i = 0; i < Config::n_layers; ++i) {
        key_cache_ptrs[i] = run_state.key_cache + i * cache_size_per_layer;
        value_cache_ptrs[i] = run_state.value_cache + i * cache_size_per_layer;
    }

    // Initialize the model
    GPT2Pretrained model(
            ptrs.wte_w,
            ptrs.wpe_w,
            ptrs.ln_f_w,
            ptrs.ln_f_b,
            ptrs.lm_head_w,
            ptrs.ln1_w,
            ptrs.ln1_b,
            ptrs.attn_c_attn_w,
            ptrs.attn_c_attn_b,
            ptrs.attn_c_proj_w,
            ptrs.attn_c_proj_b,
            ptrs.ln2_w,
            ptrs.ln2_b,
            ptrs.mlp_c_fc_w,
            ptrs.mlp_c_fc_b,
            ptrs.mlp_c_proj_w,
            ptrs.mlp_c_proj_b,
            key_cache_ptrs,
            value_cache_ptrs
    );

    using namespace std::chrono;
    auto init_start = high_resolution_clock::now();
    //Initialize sampler
    Sampler sampler(seed);
    //Initialize tokenizer
    Tokenizer tokenizer(vocab_json_path);
    //Encode tokens
    std::vector<int> input_tokens = tokenizer.encode(prompt);

    auto init_end = high_resolution_clock::now();
    auto init_time = duration_cast<duration<double, std::milli>>(init_end - init_start);
    std::cout << "Initialization time: " << init_time.count() << " ms" << std::endl;

    // Process the input tokens
    for (int token : input_tokens) {
        model.forward(token, run_state.logits);
    }
    // Generate new tokens
    std::vector<int> generated_tokens;
    generated_tokens.reserve(n_tokens_to_predict);
    int next_token = sampler.sample(run_state.logits);
    generated_tokens.push_back(next_token);

    auto first_token_end = high_resolution_clock::now();
    auto first_token_time = duration_cast<duration<double, std::milli>>(first_token_end - init_end);
    std::cout << "Time to first token: " << first_token_time.count() << " ms" << std::endl;

    for (int i = 0; i < n_tokens_to_predict; ++i) {
        model.forward(next_token, run_state.logits);
        next_token = sampler.sample(run_state.logits);
        generated_tokens.push_back(next_token);
    }

    auto last_token_end = high_resolution_clock::now();
    auto all_tokens_time = duration_cast<duration<double, std::milli>>(last_token_end - first_token_end);
    std::cout << "Time per output token: " << all_tokens_time.count()/n_tokens_to_predict << " ms" << std::endl;
    auto total_gen_time = duration_cast<duration<double, std::milli>>(last_token_end - init_end);
    std::cout << "Total generation time: " << total_gen_time.count() << " ms" << std::endl << std::endl;

    //Decode and print prompt with generated sequence
    std::string generated_text = tokenizer.decode(generated_tokens);
    std::cout << prompt << generated_text << std::endl;


    // Deallocate memory
    deallocate_run_state();
    delete[] all_weights;
    return 0;
}
