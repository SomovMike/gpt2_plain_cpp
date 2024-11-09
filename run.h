#ifndef GPT2_PLAIN_CPP_RUN_H
#define GPT2_PLAIN_CPP_RUN_H

#include <iostream>
#include <fstream>
#include <cmath> // for tanh, sqrt, pow, and M_PI
#include <cstring>
#include <algorithm>
#include <random>
#include <numeric>  // for std::accumulate
#include <chrono>
#ifdef WITH_OPENMP
#include <omp.h>
#endif
#include "utils.h"

struct Config {
    static constexpr int dim = 768;           // transformer dimension
    static constexpr int hidden_dim = 3072;   // for ffn layers (4 * dim)
    static constexpr int n_layers = 12;       // number of layers
    static constexpr int n_heads = 12;        // number of query heads (dim / 64)
    static constexpr int head_size = 64;      // one head size (dim / 12)
    static constexpr int vocab_size = 50257;  // vocabulary size
    static constexpr int seq_len = 1024;      // max sequence length
    static constexpr int topk_val = 50;       // k value for topk function
    static int n_threads;
};

struct RunState{
    //Main buffers
    float *x; // (dim)
    float *xb; // (dim) additional buffer
    //Embedding buffers
    float *emb_token_out; // (dim)
    float *emb_pos_out; // (dim)
    //Layer norm buffer
    float* ln_output; // (dim)
    //MLP buffer
    float* mlp_buffer; // (hidden_dim)
    //Transformer buffer
    float* attn_output; // (dim)
    //SelfAttention buffers
    float* qkv; // (dim*3)
    float *attn; // (n_heads, seq_len)
    float *y; // (dim)
    //logits buffer
    float *logits; // (vocab_size)
    //KV cache
    float* key_cache; // (layer, max_seq_len, n_heads, head_size)
    float* value_cache; // (layer, max_seq_len, n_heads, head_size)
} run_state;


struct WeightPointers {
    // Embedding weights
    float* wte_w;
    float* wpe_w;

    // Final LayerNorm and LM Head
    float* ln_f_w;
    float* ln_f_b;
    float* lm_head_w;

    // Transformer blocks
    float* ln1_w[Config::n_layers];
    float* ln1_b[Config::n_layers];
    float* attn_c_attn_w[Config::n_layers];
    float* attn_c_attn_b[Config::n_layers];
    float* attn_c_proj_w[Config::n_layers];
    float* attn_c_proj_b[Config::n_layers];
    float* ln2_w[Config::n_layers];
    float* ln2_b[Config::n_layers];
    float* mlp_c_fc_w[Config::n_layers];
    float* mlp_c_fc_b[Config::n_layers];
    float* mlp_c_proj_w[Config::n_layers];
    float* mlp_c_proj_b[Config::n_layers];
};

class Embedding{

private:
    float* _w;
    int _embedding_dim;
public:
    Embedding() : _w(nullptr), _embedding_dim(0) {}

    explicit Embedding(float* ptr, int embedding_dim):_w(ptr), _embedding_dim(embedding_dim){};

    void forward(int token, float* x_out);
};

class Linear{
private:
    float* _w;
    float* _b;
    int _n_input_channels;
    int _n_output_channels;
public:
    Linear() : _w(nullptr), _b(nullptr), _n_input_channels(0), _n_output_channels(0) {}

    explicit Linear(float* ptr_w, float* ptr_b, int n_input_channels,
                    int n_output_channels):_w(ptr_w), _b(ptr_b), _n_input_channels(n_input_channels),
                    _n_output_channels(n_output_channels){};

    void forward(const float* input, float* output);
};

class LayerNorm{
private:
    float* _w;
    float* _b;
    int _n_channels;
public:
    LayerNorm() : _w(nullptr), _b(nullptr), _n_channels(0) {}

    explicit LayerNorm(float* ptr_w, float* ptr_b, int n_channels):_w(ptr_w), _b(ptr_b), _n_channels(n_channels){};

    void forward(const float* input, float* output);
};


class SelfAttention{
private:
    Linear _c_attn;
    Linear _c_proj;

    float* _k_cache; // (max_seq_len, n_heads, head_size)
    float* _v_cache; // (max_seq_len, n_heads, head_size)
    int _prev_length = 0;



public:

    SelfAttention() : _k_cache(nullptr), _v_cache(nullptr), _prev_length(0) {}

    explicit SelfAttention(float* ptr_c_attn_w, float* ptr_c_attn_b, float* ptr_c_proj_w, float* ptr_c_proj_b,
                           float* ptr_k_cache, float* ptr_v_cache):
                            _c_attn(ptr_c_attn_w, ptr_c_attn_b, Config::dim, Config::dim*3),
                            _c_proj(ptr_c_proj_w, ptr_c_proj_b, Config::dim, Config::dim), _k_cache(ptr_k_cache),
                            _v_cache(ptr_v_cache){};

    void forward(const float* input, float* output);
};

class MLP{
private:
    Linear _c_fc;
    Linear _c_proj;

public:

    MLP() = default;

    explicit MLP(float* ptr_c_fc_w, float* ptr_c_fc_b, float* ptr_c_proj_w, float* ptr_c_proj_b):
                _c_fc(ptr_c_fc_w, ptr_c_fc_b, Config::dim, Config::hidden_dim),
                _c_proj(ptr_c_proj_w, ptr_c_proj_b, Config::hidden_dim, Config::dim){};

    void forward(const float* input, float* output);
};

class TransformerBlock{
private:
    LayerNorm _ln1;
    SelfAttention _attn;
    LayerNorm _ln2;
    MLP _mlp;

public:
    TransformerBlock() = default;

    explicit TransformerBlock(float* ptr_ln1_weight, float* ptr_ln1_bias, float* ptr_attn_c_attn_weight,
                              float* ptr_attn_c_attn_bias, float* ptr_attn_c_proj_weight, float* ptr_attn_c_proj_bias,
                              float* ptr_ln2_weight, float* ptr_ln2_bias, float* ptr_mlp_c_fc_weight,
                              float* ptr_mlp_c_fc_bias, float* ptr_mlp_c_proj_weight, float* ptr_mlp_c_proj_bias,
                              float* ptr_k_cache, float* ptr_v_cache):
                              _ln1(ptr_ln1_weight, ptr_ln1_bias, Config::dim),_ln2(ptr_ln2_weight, ptr_ln2_bias, Config::dim),
                              _attn(ptr_attn_c_attn_weight, ptr_attn_c_attn_bias, ptr_attn_c_proj_weight, ptr_attn_c_proj_bias,
                                    ptr_k_cache, ptr_v_cache),
                              _mlp(ptr_mlp_c_fc_weight, ptr_mlp_c_fc_bias, ptr_mlp_c_proj_weight, ptr_mlp_c_proj_bias){};

    void forward(const float* input, float* output);

};

class GPT2Pretrained{
private:
    //Layers
    Embedding _wte;
    Embedding _wpe;
    TransformerBlock _h[Config::n_layers];
    LayerNorm _ln_f;
    Linear _lm_head;

    //KV-cache related
    int _prev_length = 0;

public:
    GPT2Pretrained(
            float* ptr_wte_w,
            float* ptr_wpe_w,
            float* ptr_ln_f_weight,
            float* ptr_ln_f_bias,
            float* ptr_lm_head_w,
            // Arrays of pointers for Transformer Blocks
            float* ptr_h_ln1_weight[Config::n_layers],
            float* ptr_h_ln1_bias[Config::n_layers],
            float* ptr_h_attn_c_attn_weight[Config::n_layers],
            float* ptr_h_attn_c_attn_bias[Config::n_layers],
            float* ptr_h_attn_c_proj_weight[Config::n_layers],
            float* ptr_h_attn_c_proj_bias[Config::n_layers],
            float* ptr_h_ln2_weight[Config::n_layers],
            float* ptr_h_ln2_bias[Config::n_layers],
            float* ptr_h_mlp_c_fc_weight[Config::n_layers],
            float* ptr_h_mlp_c_fc_bias[Config::n_layers],
            float* ptr_h_mlp_c_proj_weight[Config::n_layers],
            float* ptr_h_mlp_c_proj_bias[Config::n_layers],
            float* ptr_key_cache[Config::n_layers],
            float* ptr_value_cache[Config::n_layers]
    ) : _wte(ptr_wte_w, Config::dim),
        _wpe(ptr_wpe_w, Config::dim),
        _ln_f(ptr_ln_f_weight, ptr_ln_f_bias, Config::dim),
        _lm_head(ptr_lm_head_w, nullptr, Config::dim, Config::vocab_size) {

        // Initialize Transformer Blocks
        for (int i = 0; i < Config::n_layers; ++i) {
            _h[i] = TransformerBlock(
                    ptr_h_ln1_weight[i], ptr_h_ln1_bias[i],
                    ptr_h_attn_c_attn_weight[i], ptr_h_attn_c_attn_bias[i],
                    ptr_h_attn_c_proj_weight[i], ptr_h_attn_c_proj_bias[i],
                    ptr_h_ln2_weight[i], ptr_h_ln2_bias[i],
                    ptr_h_mlp_c_fc_weight[i], ptr_h_mlp_c_fc_bias[i],
                    ptr_h_mlp_c_proj_weight[i], ptr_h_mlp_c_proj_bias[i],
                    ptr_key_cache[i], ptr_value_cache[i]
            );
        }
    }

public:

    void forward(int token, float* logits);
};

class Sampler{

private:
    std::mt19937 _rd;

    static void _topk(float* input, int input_size, int k, std::vector<float>& top_values, std::vector<int>& top_indices);
    size_t _multinomial_sample(const std::vector<float>& probabilities);

public:

    explicit Sampler(int seed = -1){
        if (seed != -1)
            _rd.seed(seed);
        else{
            std::random_device rd;
            _rd.seed(rd());  // Seed with a random value
        }
    }

    int sample(float* logits);
};

#endif //GPT2_PLAIN_CPP_RUN_H
