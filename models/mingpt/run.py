import os
import sys
import torch
import json
from torch.utils.data import random_split
from model import GPT, GPTConfig, OptimizerConfig, create_optimizer
from char_dataset import CharDataset, FakeDataset, DataConfig
from omegaconf import DictConfig, OmegaConf
import hydra

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_path)

from common import Trainer, TrainerConfig, measurement


def get_train_objs(gpt_cfg: GPTConfig, opt_cfg: OptimizerConfig, data_cfg: DataConfig):
    if data_cfg.vocab_size is not None:
        dataset = FakeDataset(data_cfg)
    else:
        dataset = CharDataset(data_cfg)
    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size
    model = GPT(gpt_cfg)
    model = model.to(dtype=torch.bfloat16)
    optimizer = create_optimizer(model, opt_cfg)

    return model, optimizer, train_set, test_set

# @hydra.main(config_path=".", config_name="config")
# def main(cfg: DictConfig):
#     print(OmegaConf.to_yaml(cfg))

@hydra.main(version_base=None, config_path=".", config_name="gpt2_train_cfg")
def main(cfg: DictConfig):

    gpt_cfg = GPTConfig(**cfg["model"]["gpt_config"])
    opt_cfg = OptimizerConfig(**cfg["optimizer_config"])
    data_cfg = DataConfig(**cfg["data_config"])
    trainer_cfg = TrainerConfig(**cfg["trainer_config"])

    model, optimizer, train_data, test_data = get_train_objs(gpt_cfg, opt_cfg, data_cfg)
    trainer = Trainer(trainer_cfg, model, optimizer, train_data, test_data)
    trainer.train()

    measurement_data = trainer.get_measurement()
    measurement_data.batch_size = measurement_data.batch_size * gpt_cfg.block_size
    measurement_data.model = gpt_cfg.model_type
    measurement_data.model_size = model.get_size()
    measurement_data.model_params = model.n_params

    measurement.save_dataclass_to_json(measurement_data, gpt_cfg.model_type, "./gpt_measurement/")


if __name__ == "__main__":
    main()
