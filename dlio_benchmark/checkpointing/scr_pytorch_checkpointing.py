"""
Copyright (c) 2025, UChicago Argonne, LLC
All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import torch

from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.utils.utility import Profile

from dlio_benchmark.common.constants import MODULE_CHECKPOINT
from dlio_benchmark.common.enumerations import CheckpointLocationType
from dlio_benchmark.utils.utility import DLIOMPI
import logging
import scr


def get_torch_datatype(datatype):
    if datatype == "fp32":
        return torch.float32
    elif datatype == "fp16":
        return torch.float16
    elif datatype == "fp64":
        return torch.float64
    elif datatype == "int8":
        return torch.int8
    elif datatype == "uint8":
        return torch.uint8
    elif datatype == "bf16":  # bfloat16
        return torch.bfloat16
    else:
        raise Exception(f"Invalid datatype {datatype}")


dlp = Profile(MODULE_CHECKPOINT)


class SCRPyTorchCheckpointing(BaseCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """Static access method."""
        if SCRPyTorchCheckpointing.__instance is None:
            SCRPyTorchCheckpointing.__instance = SCRPyTorchCheckpointing()
        return SCRPyTorchCheckpointing.__instance

    @dlp.log_init
    def __init__(self):
        super().__init__("pt")
        scr.init()

    @dlp.log
    def get_tensor(self, length, datatype="int8"):
        return torch.ones(length, dtype=get_torch_datatype(datatype))

    @dlp.log
    def save_state(self, suffix, state, fsync=False):
        name = self.get_name(suffix)
        scr_name = scr.route_file(name)
        logging.debug(f"SCR checkpointing on file {scr_name} for {name}")
        with open(scr_name, "wb") as f:
            torch.save(state, f)
            if fsync:
                os.fsync(f.fileno())

    @dlp.log
    def load_state(self, suffix, state):
        name = self.get_name(suffix)
        scr_name = scr.route_file(name)
        state = dict()  # clear up
        state = torch.load(name)
        self.logger.debug(f"checkpoint state loaded: {state}")
        assert len(state.keys()) > 0

    @dlp.log
    def save_checkpoint(self, epoch, step_number):
        with Profile(
            name=f"checkpoint_start_{epoch}_{step_number}",
            cat=MODULE_CHECKPOINT,
            epoch=epoch,
            step=step_number,
        ):
            scr.start_output(f"scr-chk-{epoch}-{step_number}", scr.FLAG_CHECKPOINT)
        valid = True
        try:
            if DLIOMPI.get_instance().rank() == 0:
                logging.debug(
                    f"SCR checkpointing for epoch:{epoch} and step:{step_number}"
                )
            super().save_checkpoint(epoch, step_number)
        except:
            # failed to write file
            valid = False
        with Profile(
            name=f"checkpoint_end_{epoch}_{step_number}",
            cat=MODULE_CHECKPOINT,
            epoch=epoch,
            step=step_number,
        ) as prof:
            rc = scr.complete_output(valid)
            prof.update(args={"valid": str(valid)})

    @dlp.log
    def load_checkpoint(self, epoch, step_number):
        super().load_checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        scr.finalize()
