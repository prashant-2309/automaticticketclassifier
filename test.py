import mmap
import torch
import json
import os
from huggingface_hub import hf_hub_download


def load_file(filename, device):
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")
            metadata_bytes = m.read(n)
            metadata = json.loads(metadata_bytes)

    size = os.stat(filename).st_size
    storage = torch.ByteStorage.from_file(filename, shared=False, size=size).untyped()
    offset = n + 8
    return {name: create_tensor(storage, info, offset) for name, info in metadata.items() if name != "__metadata__"}


DTYPES = {"F32": torch.float32}
device = "cpu"


def create_tensor(storage, info, offset):
    dtype = DTYPES[info["dtype"]]
    shape = info["shape"]
    start, stop = info["data_offsets"]
    return torch.asarray(storage[start + offset : stop + offset], dtype=torch.uint8).view(dtype=dtype).reshape(shape)


def main():
    filename = hf_hub_download("gpt2", filename="C:\Users\w191661\OneDrive - Worldline SA\Documents\Hackathon\BERT_Categorizer\model.safetensors")
    weights = load_file(filename, device)
    print(weights.keys())


if __name__ == "__main__":
    main()