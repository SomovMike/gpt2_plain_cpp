#ifndef GPT2_PLAIN_CPP_UTILS_H
#define GPT2_PLAIN_CPP_UTILS_H

#include <iostream>
#include <map>
#include <regex>
#include <fstream>
#include <random>


class Tokenizer{
private:
    using id    = int;
    using token = std::string;

    std::map<token, id> _token_to_id;
    std::map<id, token> _id_to_token;

    std::string _replace(const std::string & s, const std::string & from, const std::string & to);
    void _json_parse(const std::string & fname);
    void _split_words(std::string str, std::vector<std::string>& words);

public:

    explicit Tokenizer(const std::string & fname);

    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
};

void storeFloatsToFile(const float* arr, int n, const std::string& filename);

float* read_checkpoint(const std::string& filename, size_t& total_params);

#endif //GPT2_PLAIN_CPP_UTILS_H
