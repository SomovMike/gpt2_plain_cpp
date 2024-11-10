FROM ubuntu:latest

RUN apt-get update && apt-get install -y git \
    python3 \
    wget \
    cmake \
    clang \
    python3-pip

# Clone the repository into a specific directory in the container
RUN git clone https://github.com/SomovMike/gpt2_plain_cpp.git /app
# Set the working directory
WORKDIR /app

RUN wget https://huggingface.co/openai-community/gpt2/resolve/main/pytorch_model.bin
RUN wget https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
RUN pip install --no-cache-dir --break-system-packages torch numpy

