from collections import OrderedDict
import time
import torch

from torchsnapshot import StateDict, Snapshot as Snapshot
import warnings
import copy
from typing import Dict, List
from multiprocessing import Pool
import multiprocessing as mp

warnings.filterwarnings("ignore")

def color_string(text):
    return f"\033[94m{text}\033[00m"


class Model(torch.nn.Module):
    def __init__(self, param_size: int, num_params: int, cpu=False) -> None:
        super().__init__()
        self.param_size = param_size
        for i in range(num_params):
            self.register_parameter(
                f"param_{i}",
                torch.nn.Parameter(
                    torch.rand(
                        int(param_size),
                        device='cpu' if cpu else torch.cuda.current_device(),
                        dtype=torch.float32,
                    )
                ),
            )


def memcpy(model):
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()

    best_model_state_dict = {k: v.to("cpu") for k, v in model.state_dict().items()}
    best_model_state_dict = OrderedDict(best_model_state_dict)
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender)


def move(model):
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()

    best_model_state_dict = model.to("cpu")

    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender)

def move_prealloc(model):
    # This is stupid lazy way of doing it. The move_prealloc_copy is better
    # and more generic
    model_cpu = copy.deepcopy(model.to('cpu'))
    model.to(torch.cuda.current_device())
    for _, tensor in model_cpu.state_dict().items(): 
        tensor.zero_()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()

    best_model_state_dict = model_cpu.load_state_dict(model.state_dict())

    ender.record()
    torch.cuda.synchronize()
    # print(model_cpu.state_dict())
    # print(model.state_dict())
    return starter.elapsed_time(ender)

def extract_tensor_frame(m) -> List[Dict]:
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays, and return the stripped model and tensors.
    """
    tensors = {}
    for k, v in m.items():
        # Store the tensors in Python dictionaries
        tensors[k] = torch.zeros(v.shape, dtype=v.dtype, device='cpu')
    # Make a copy of the original model and strip all tensors and
    # buffers out of the copy.
    # Make sure the copy is configured for inference.
    return tensors


def assign_tensors(m, existing_dict) -> List[Dict]:
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays, and return the stripped model and tensors.
    """
    for k, v in m.items():
        # Store the tensors in Python dictionaries
        existing_dict[k].copy_(v)
    
def move_prealloc_copy(model):
    model_cpu = extract_tensor_frame(model.state_dict())
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    best_model_stateict = assign_tensors(model.state_dict(), model_cpu)
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender)


def async_snapshot(model):
    start = time.time()
    app_state = {"model": model}
    Snapshot.async_take(
        path="./snapshot/",
        app_state=app_state,
    )
    end = time.time()
    return end - start


def snapshot(model):
    start = time.time()
    app_state = {"model": model}
    Snapshot.take(
        path="./snapshot/",
        app_state=app_state,
    )
    end = time.time()
    return end - start


if __name__ == "__main__":
    mp.set_start_method('spawn')
    functions = [move_prealloc_copy]
    for f in functions:
        # for i in range(1, 2):
        model = Model(param_size=2147483647 * 4, num_params=1)

        sz = sum(t.nelement() * t.element_size() for t in model.parameters())
        sz_gb = sz / 2**30

        time_taken = f(model)

        print(
            f"Model Size {color_string(sz_gb)}, Snapshot with {color_string(f.__name__)}, time {color_string(time_taken)}"
        )
