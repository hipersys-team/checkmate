import torch
import torchvision
import multiprocessing
import io
import os
import sys
import signal
import restore
import time

signal.signal(signal.SIGINT, signal.SIG_DFL)


def run_training(model, optimizer):
    from torch.nn.parallel import DistributedDataParallel as DDP

    start = time.time()
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(0)
    backend = time.time()
    model.to(0)
    model = DDP(model)
    print(f"Time taken to initialize Backend: ", {(backend - start) * 1000})
    print(f"Time taken to initialize DDP: ", {(time.time() - backend) * 1000})
    print("Training completed")
    torch.distributed.destroy_process_group()


def bytes_to_model(bytes):
    bytes.seek(0)
    return torch.load(bytes, weights_only=False)


def fetch_model(queue, model_size):
    client = restore.Client(5678, 4)
    buffer_tensor = torch.zeros(model_size, dtype=torch.uint8)
    start = time.time()
    client.receive(buffer_tensor.data_ptr(), model_size)
    print(f"Time taken to receive model: ", {(time.time() - start) * 1000})
    buffer = buffer_tensor.numpy().tobytes()
    queue.put(buffer)


def get_training_args(model_name, model_size):
    rank = int(os.environ["RANK"])
    if rank == 0:
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        process = multiprocessing.Process(
            target=fetch_model,
            args=(
                queue,
                model_size,
            ),
        )
        process.start()
        process.join()

        bytes = queue.get()
        received_buffer = io.BytesIO(bytes)
        start = time.time()
        app_state = bytes_to_model(received_buffer)
        model = app_state["model"]
        optimizer = app_state["optimizer"]
        print(f"Time taken to load model: ", {(time.time() - start) * 1000})
    else:
        if model_name.startswith("gpt") or "B" in model_name:
            parent_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../storage/models/llama/")
            )
            sys.path.insert(0, parent_path)
            from llama.llama_config import all_models
            from llama.llama_model import Transformer

            model_config = all_models["llama2"][args.model]
            model = Transformer.from_model_args(model_config).to(
                "cpu", dtype=torch.bfloat16
            )
        else:
            model = torchvision.models.__dict__[args.model]()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    return model, optimizer


if __name__ == "__main__":
    import argparse

    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=[
            "resnet50",
            "resnet152",
            "vit_h_14",
            "gpt2",
            "gpt3xl",
            "gpt3_6_7B",
            "7B",
            "13B",
        ],
        type=str,
        help="model",
    )
    args = parser.parse_args()
    model_size = {
        "resnet50": 307116974,
        "resnet152": 723555392,
        "vit_h_14": 7584984310,
        "gpt2": 9472984822,
        "gpt3xl": 8060191498,
        "gpt3_6_7B": 40279812722,
    }

    model, optimizer = get_training_args(args.model, model_size[args.model])
    run_training(model, optimizer)
