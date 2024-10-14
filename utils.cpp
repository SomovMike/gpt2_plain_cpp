#include "utils.h"

Tokenizer::Tokenizer(const std::string &fname) {
    // fill _token_to_id
    _json_parse(fname);

    for (const auto & kv : _token_to_id) {
        _id_to_token[kv.second] = kv.first;
    }
}


std::string Tokenizer::_replace(const std::string &s, const std::string &from, const std::string &to) {
    std::string result = s;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}


void Tokenizer::_json_parse(const std::string &fname) {
    // read file into string
    std::string json;
    {
        std::ifstream ifs(fname);
        if (!ifs) {
            std::cerr << "Failed to open %s\n", fname.c_str();
            exit(1);
        }

        json = std::string((std::istreambuf_iterator<char>(ifs)),
                           (std::istreambuf_iterator<char>()));
    }

    // parse json
    {
        bool has_key  = false;
        bool in_token = false;

        std::string str_key;
        std::string str_val;

        u_int n = json.size();
        for (int i = 1; i < n; ++i) {
            if (!in_token) {
                if (json[i] == ' ') continue;
                if (json[i] == '"') {
                    in_token = true;
                    continue;
                }
            } else {
                if (json[i] == '\\' && i+1 < n) {
                    if (!has_key) {
                        str_key += json[i];
                    } else {
                        str_val += json[i];
                    }
                    ++i;
                } else if (json[i] == '"') {
                    if (!has_key) {
                        has_key = true;
                        ++i;
                        while (json[i] == ' ') ++i;
                        ++i; // :
                        while (json[i] == ' ') ++i;
                        if (json[i] != '\"') {
                            while (json[i] != ',' && json[i] != '}') {
                                str_val += json[i++];
                            }
                            has_key = false;
                        } else {
                            in_token = true;
                            continue;
                        }
                    } else {
                        has_key = false;
                    }

                    str_key = _replace(str_key, "\\u0120", " " ); // \u0120 -> space
                    str_key = _replace(str_key, "\\u010a", "\n"); // \u010a -> new line
                    str_key = _replace(str_key, "\\\"",    "\""); // \\\"   -> "

                    _token_to_id[str_key] = std::stoi(str_val);

                    str_key = "";
                    str_val = "";
                    in_token = false;
                    continue;
                }
                if (!has_key) {
                    str_key += json[i];
                } else {
                    str_val += json[i];
                }
            }
        }
    }
}

void Tokenizer::_split_words(std::string str, std::vector<std::string> &words) {
    const std::string pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
    const std::regex re(pattern);
    std::smatch m;

    while (std::regex_search(str, m, re)) {
        for (auto x : m) {
            words.push_back(x);
        }
        str = m.suffix();
    }
}

std::vector<int> Tokenizer::encode(const std::string &text) {
    std::vector<std::string> words;
    // first split the text into words
    _split_words(text, words);

    // find the longest token that forms each word in words:
    std::vector<int> tokens;
    for (const auto & word : words) {
        for (int i = 0; i < (int) word.size(); ){
            for (int j = (int) word.size() - 1; j >= i; j--){
                auto cand = word.substr(i, j-i+1);
                auto it = _token_to_id.find(cand);
                if (it != _token_to_id.end()){ // word.substr(i, j-i+1)
                    tokens.push_back(it->second);
                    i = j + 1;
                    break;
                }
            }
        }
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<int> & tokens) {
    std::string result;

    for (const auto& token : tokens) {
        // Find the token in the id_to_token map
        auto it = _id_to_token.find(token);
        if (it != _id_to_token.end()) {
            result += it->second;  // Append the corresponding token text to the result
        }
    }

    return result;
}

void storeFloatsToFile(const float* arr, int n, const std::string& filename) {
    std::ofstream outFile(filename);  // Open file for writing

    if (!outFile) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    for (int i = 0; i < n; ++i) {
        outFile << arr[i] << std::endl;  // Write each float to the file
    }

    outFile.close();  // Close the file
}


float* read_checkpoint(const std::string& filename, size_t& total_params) {
    std::ifstream infile(filename, std::ios::binary | std::ios::ate);
    if (!infile) {
        std::cerr << "Error opening weights file: " << filename << std::endl;
        return nullptr;
    }

    // Get the size of the file
    std::streamsize file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    // Calculate the total number of floats
    total_params = file_size / sizeof(float);

    // Allocate memory for the weights
    float* weights = new float[total_params];

    // Read the weights into the array
    if (!infile.read(reinterpret_cast<char*>(weights), file_size)) {
        std::cerr << "Error reading weights from file: " << filename << std::endl;
        delete[] weights;
        return nullptr;
    }

    infile.close();
    return weights;
}

