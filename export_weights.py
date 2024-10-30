import struct
import torch
import argparse

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def export(model_weights, filepath):
    out_file = open(filepath, 'wb')

    for tensor in model_weights.values():
        if tensor.size() == torch.Size([1, 1, 1024, 1024]):
            continue
        if (tensor.size() == torch.Size([768, 2304]) or tensor.size() == torch.Size([768, 3072])
                or tensor.size() == torch.Size([3072, 768]) or tensor.size() == torch.Size([768, 768])):
            tensor = tensor.T.contiguous()
        serialize_fp32(out_file, tensor)

    decode_emb = model_weights['wte.weight']
    serialize_fp32(out_file, decode_emb)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("checkpoint", type=str, help="model checkpoint, .bin file")
    args = parser.parse_args()
    model_weights = torch.load(args.checkpoint)

    export(model_weights, filepath=args.filepath)

