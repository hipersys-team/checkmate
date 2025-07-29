from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
import json
import os
from datetime import datetime

# class SnapshotPolicy(Enum):
#     ASYNC = 0
#     SYNC = 1
#     DISABLED = 2

@dataclass
class MeasurementData:
    snap_policy: str
    total_iter_time: float
    total_snap_time: float
    total_iter_cnt: int
    total_snap_cnt: float
    snap_overhead: float
    model: str = ""
    batch_size: int = 0
    world_size: int = 0
    model_size: float = 0 # MB
    model_params: int = 0

@dataclass
class MeasurementConfig:
  path: str

def save_dataclass_to_json(data, model_name: str, path: str = './'):
    """
    Saves a dataclass instance to a JSON file with a filename based on the current timestamp and model name.

    Parameters:
    - data: The dataclass instance to save.
    - model_name: The name of the model to include in the filename.
    - path: The directory path where the file will be saved. Defaults to the current directory.

    The filename will have the format YYYYMMDD_HHMMSS_model_name.json.
    """
    if not is_dataclass(data):
        raise ValueError("The data parameter must be a dataclass instance.")

    # Get current time
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")

    filename = f"{formatted_time}_{model_name}.json"
    full_path = path + filename

    # Convert dataclass to dictionary
    data_dict = asdict(data)

    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    # Save to JSON file
    with open(full_path, 'w') as f:
        json.dump(data_dict, f, indent=4)

    print(f"Data saved to {full_path}")
