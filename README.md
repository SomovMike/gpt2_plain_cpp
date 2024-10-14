# GPT-2 inference in Plain C++
This project is a simple implementation of
OpenAI’s GPT-2 (124M parameters) language model inference written entirely 
in plain C++ for CPU (unfortunately I don't have any GPU). The main goal is to create a 
self-contained GPT-2 inference engine without 
relying on external libraries or dependencies. 
By building everything from scratch, I want to gain a 
deeper understanding of how LLMs works and have 
full control over performance optimizations. The initial focus is on correctness 
and simplicity. Performance optimizations will be 
introduced incrementally, aiming to enhance 
inference speed and efficiency.

## Getting Started

### Requirements

- **C++ compiler:** A compiler that supports C++17 or higher (e.g., GCC, Clang, MSVC).
- **CMake:** Version 3.15 or higher for building the project.
- **Python3 with Pytorch installed:** For running the conversion script to prepare model weights.

### Preparing the Model Weights and Vocabulary

Obtain the GPT-2 model weights (pytorch_model.bin) from [Hugging Face](https://huggingface.co/openai-community/gpt2/tree/main).
The C++ code expects the model weights in a specific binary
format. Use the provided [export_weights.py](export_weights.py) script
to convert the pytorch_model.bin file into a binary file containing all weights serialized as 32-bit floats:
```console
python export_weights.py <output_filepath> <pytorch_model.bin_filepath>
``` 

Then you also need to download [vocab.json](https://huggingface.co/openai-community/gpt2/blob/main/vocab.json)
from the same Hugging Face repo.

### Build

```console
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
``` 

### Running the inference

After building the executable, try an example:
```console
./gpt2_plain_cpp <model_weights_path> <vocab_json_path> -p "I'm learning deep nueral networks" -n 20
```

The following options are supported:
```
Required:
[model_weights_path] -- path to .bin file generated with export_weights.py script
[vocab_json_path] -- path to vocab.json for GPT2
Optional:
[-p <std::string>] prompt to start generation with
[-n <int>] number of tokens to generate
[-s <int>] seed if you want to reproduce the result
```

## Optimization path

Here I will try to describe all performance optimizations 
I made so far. Our main metrics will be Time To First Token (TTFT) and
Time Per Output Token (TPOT) and Total generation time (TTFT + TPOT x number of generated tokens).

|                | TTFT (50 tokens promt) | TPOT (200 tokens in total) | Total generation time |
|----------------|------------------------|----------------------------|-----------------------|
| Optimization 0 | 7900 ms                | 155 ms                     | 38.9s                 |
| Optimization 1 |                        |                            |                       |
| Optimization 2 |                        |                            |                       |

### 0. No optimizations

Initially I implemented very naive inference without any optimizations (except KV-cache).


### 1. 





