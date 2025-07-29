import torch
import torchvision
import io
import os
import sys
import signal
import restore
import time

signal.signal(signal.SIGINT, signal.SIG_DFL)


def optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if param.requires_grad:
                if param not in optimizer.state:
                    optimizer.state[param] = {
                        "step": 0,
                        "exp_avg": torch.zeros_like(param.data),
                        "exp_avg_sq": torch.zeros_like(param.data),
                    }
            optimizer.state[param]["exp_avg"].uniform_(-0.1, 0.1)
            optimizer.state[param]["exp_avg_sq"].uniform_(0, 0.01)
    return optimizer


if __name__ == "__main__":
    import argparse

    parent_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../storage/models/llama/")
    )
    sys.path.insert(0, parent_path)
    from llama.llama_config import all_models
    from llama.llama_model import Transformer

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

    if args.model.startswith("gpt") or "B" in args.model:
        model_config = all_models["llama2"][args.model]
        model = Transformer.from_model_args(model_config).to(
            "cpu", dtype=torch.bfloat16
        )
    else:
        model = torchvision.models.__dict__[args.model]()

    optimizer = optimizer(model)
    app_state = {
        "model": model,
        "optimizer": optimizer,
    }
    buffer = io.BytesIO()
    start = time.time()
    torch.save(app_state, buffer)
    print(f"Time taken to save model: ", {(time.time() - start) * 1000})
    buffer.seek(0)
    data = buffer.read()
    print(len(data))

    restore = restore.Server(5678, 4)
    restore.send(data)
    print("Model sent")
