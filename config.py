# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for parsing robomimic configuration files."""


import os

XARM_ENVS_DATA_DIR = "robomimic_config"

ROBOMIMIC_CONFIG_FILES_DICT = {
    "XArm7Env-v1": {
        "bc": os.path.join(XARM_ENVS_DATA_DIR, "pick_and_place_bc.json"),
        "bcq": os.path.join(XARM_ENVS_DATA_DIR, "pick_and_place_bcq.json"),
    }
}

"""Mapping from environment names to imitation learning config files."""

__all__ = ["ROBOMIMIC_CONFIG_FILES_DICT"]
