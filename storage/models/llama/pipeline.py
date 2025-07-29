# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

from torch.distributed.pipelining.schedules import (
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    Schedule1F1B,
)

import copy
from typing import Callable, Union

import torch
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage

from llama.llama_model import ModelArgs


def generate_split_points(pp_dim, model_config: ModelArgs):
    schedule_class = Schedule1F1B
    if issubclass(schedule_class, PipelineScheduleSingle):
        num_stages_per_rank = 1
    elif issubclass(schedule_class, PipelineScheduleMulti):
        # Multi-stage schedules support more than 2 stages per rank, but this is the default if
        # no pipeline split is specified
        num_stages_per_rank = 2
    else:
        raise ValueError(f"Unsupported pipeline schedule")
    total_stages = pp_dim * num_stages_per_rank
    num_layers = model_config.n_layers
    if total_stages > num_layers:
        raise ValueError("Total stages cannot be greater than the number of layers")

    base_interval = num_layers // total_stages
    extra_layers = num_layers % total_stages

    splits = []
    current_layer = 0
    for i in range(total_stages - 1):
        if i == 0:
            current_layer += base_interval
        else:
            # Middle stages get an extra layer if there are any remaining
            if extra_layers > 0:
                current_layer += base_interval + 1
                extra_layers -= 1
            else:
                current_layer += base_interval
        splits.append("layers." + str(current_layer))
    print(
        f"No 'pipeline_parallel_split_points' so the generated splits are: {splits} \
This may be sub-optimal as the number of layers per stage may be unbalanced."
    )
    return splits


# TODO(whc) should this be a utility inside torch.pipelining?
def stage_ids_this_rank(
    pp_rank: int, pp_size: int, num_stages: int, style: str = "loop"
) -> Tuple[int]:
    """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
    assert (
        num_stages % pp_size == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
    stages_per_rank = num_stages // pp_size
    if style == "loop":
        return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
    elif style == "v":
        assert (
            stages_per_rank == 2
        ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(
            zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1))
        )
        return stage_v_pairs[pp_rank]


DeviceType = Union[int, str, torch.device]

def pipeline_llama(
    model: nn.Module,
    pp_size: int,
    pp_rank: int,
    model_config: ModelArgs,
):
    models = pipeline_llama_manual_split(model, pp_size, pp_rank, model_config)
    return models

def pipeline_llama_manual_split(
    whole_model: nn.Module,
    pp_size: int,
    pp_rank: int,
    model_config: ModelArgs,
):
    """
    This API extracts one torch.nn.Module objects for the part of the model configured to run inside this stage.

    It wraps the model chunk in a ManualPipelineStage object and returns both the stage and model objects.

    The stage object is used to create a pipeline schedule, and the model object can be used for applying SPMD
    parallelism.
    """
    splits = generate_split_points(pp_size, model_config)

    def _build_stage(stage_idx, start_layer, stop_layer, is_first=False, is_last=False):
        model = copy.deepcopy(whole_model)
        if not is_first:
            model.tok_embeddings = None

        drop_layers = start_layer is not None
        for name in list(model.layers.keys()):
            # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
            if f"layers.{name}" == start_layer:
                drop_layers = False
            if f"layers.{name}" == stop_layer:
                drop_layers = True
            if drop_layers:
                del model.layers[name]

        if not is_last:
            model.norm = None
            model.output = None
        return model

    num_stages = len(splits) + 1
    stage_idx = pp_rank

    models = []
    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style="loop"):
        start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        model_chunk = _build_stage(
            stage_idx,
            start_layer,
            stop_layer,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        print(
            f"PP rank {pp_rank} is building stage_idx {stage_idx}"
            f" with start_layer {start_layer}, stop_layer {stop_layer}: model chunk \n{model_chunk}"
        )
        models.append(model_chunk)
    return models
