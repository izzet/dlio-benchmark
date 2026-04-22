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
import sys
from datetime import datetime
import logging
from time import time, sleep as base_sleep
from functools import wraps
import threading
import json
import socket
import argparse

import psutil
import numpy as np

from dlio_benchmark.common.enumerations import MPIState
from dftracer.python import (
    dftracer as PerfTrace,
    dft_fn as Profile,
    ai as dft_ai,
    DFTRACER_ENABLE
)

LOG_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
_FETCH_DELAY_CFG = None
_FETCH_DELAY_LOGGED_PIDS = set()

OUTPUT_LEVEL = 35
logging.addLevelName(OUTPUT_LEVEL, "OUTPUT")
def output(self, message, *args, **kwargs):
    if self.isEnabledFor(OUTPUT_LEVEL):
        self._log(OUTPUT_LEVEL, message, args, **kwargs)
logging.Logger.output = output

class DLIOLogger:
    __instance = None

    def __init__(self):
        self.logger = logging.getLogger("DLIO")
        #self.logger.setLevel(logging.DEBUG)
        if DLIOLogger.__instance is not None:
            raise Exception(f"Class {self.classname()} is a singleton!")
        else:
            DLIOLogger.__instance = self
    @staticmethod
    def get_instance():
        if DLIOLogger.__instance is None:
            DLIOLogger()
        return DLIOLogger.__instance.logger
    @staticmethod
    def reset():
        DLIOLogger.__instance = None
# MPI cannot be initialized automatically, or read_thread spawn/forkserver
# child processes will abort trying to open a non-existant PMI_fd file.
import mpi4py
p = psutil.Process()


def add_padding(n, num_digits=None):
    str_out = str(n)
    if num_digits != None:
        return str_out.rjust(num_digits, "0")
    else:
        return str_out


def utcnow(format=LOG_TS_FORMAT):
    return datetime.now().strftime(format)


# After the DLIOMPI singleton has been instantiated, the next call must be
# either initialize() if in an MPI process, or set_parent_values() if in a
# non-MPI pytorch read_threads child process.
class DLIOMPI:
    __instance = None

    def __init__(self):
        if DLIOMPI.__instance is not None:
            raise Exception(f"Class {self.classname()} is a singleton!")
        else:
            self.mpi_state = MPIState.UNINITIALIZED
            DLIOMPI.__instance = self

    @staticmethod
    def get_instance():
        if DLIOMPI.__instance is None:
            DLIOMPI()
        return DLIOMPI.__instance

    @staticmethod
    def reset():
        DLIOMPI.__instance = None

    @classmethod
    def classname(cls):
        return cls.__qualname__

    def initialize(self):
        from mpi4py import MPI
        if self.mpi_state == MPIState.UNINITIALIZED:
            # MPI may have already been initialized by dlio_benchmark_test.py
            if not MPI.Is_initialized():
                MPI.Init()
            
            self.mpi_state = MPIState.MPI_INITIALIZED
            split_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
            self.mpi_local_comm = split_comm
            # Number of processes on this node and local rank
            local_ppn = split_comm.size
            self.mpi_local_rank = split_comm.rank
            # Create a communicator of one leader per node
            if split_comm.rank == 0:
                leader_comm = MPI.COMM_WORLD.Split(color=0, key=MPI.COMM_WORLD.rank)
                # Gather each node's process count
                ppn_list = leader_comm.allgather(local_ppn)
            else:
                # Non-leaders do not participate
                MPI.COMM_WORLD.Split(color=MPI.UNDEFINED, key=MPI.COMM_WORLD.rank)
                ppn_list = None
            # Broadcast the per-node list to all processes
            self.mpi_ppn_list = MPI.COMM_WORLD.bcast(ppn_list, root=0)
            # Total number of nodes
            self.mpi_nodes = len(self.mpi_ppn_list)
            # Total world size and rank
            self.mpi_size = MPI.COMM_WORLD.size
            self.mpi_rank = MPI.COMM_WORLD.rank
            self.mpi_world = MPI.COMM_WORLD
            # Compute node index and per-node offset
            offsets = [0] + list(np.cumsum(self.mpi_ppn_list)[:-1])
            # Determine which node this rank belongs to
            for idx, off in enumerate(offsets):
                if self.mpi_rank >= off and self.mpi_rank < off + self.mpi_ppn_list[idx]:
                    self.mpi_node = idx
                    break
            os.environ["DLIO_MPI_NODE_INDEX"] = str(self.mpi_node)
            os.environ["DLIO_MPI_LOCAL_RANK"] = str(self.mpi_local_rank)
            os.environ["DLIO_MPI_NUM_NODES"] = str(self.mpi_nodes)
            os.environ["DLIO_MPI_GLOBAL_RANK"] = str(self.mpi_rank)
            os.environ["DLIO_MPI_HOSTNAME"] = socket.gethostname()
        elif self.mpi_state == MPIState.CHILD_INITIALIZED:
            raise Exception(f"method {self.classname()}.initialize() called in a child process")
        else:
            pass    # redundant call

    # read_thread processes need to know their parent process's rank and comm_size,
    # but are not MPI processes themselves.
    def set_parent_values(self, parent_rank, parent_comm_size):
        if self.mpi_state == MPIState.UNINITIALIZED:
            self.mpi_state = MPIState.CHILD_INITIALIZED
            self.mpi_rank = parent_rank
            self.mpi_size = parent_comm_size
            self.mpi_world = None
            self.mpi_local_comm = None
            self.mpi_local_rank = int(os.environ.get("DLIO_MPI_LOCAL_RANK", "0") or 0)
            self.mpi_node = int(os.environ.get("DLIO_MPI_NODE_INDEX", "0") or 0)
            self.mpi_nodes = int(os.environ.get("DLIO_MPI_NUM_NODES", "1") or 1)
            self.mpi_ppn_list = []
            os.environ["DLIO_MPI_GLOBAL_RANK"] = str(parent_rank)
        elif self.mpi_state == MPIState.MPI_INITIALIZED:
            raise Exception(f"method {self.classname()}.set_parent_values() called in a MPI process")
        else:
            raise Exception(f"method {self.classname()}.set_parent_values() called twice")

    def rank(self):
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.rank() called before initializing MPI")
        else:
            return self.mpi_rank

    def size(self):
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.size() called before initializing MPI")
        else:
            return self.mpi_size

    def comm(self):
        if self.mpi_state == MPIState.MPI_INITIALIZED:
            return self.mpi_world
        elif self.mpi_state == MPIState.CHILD_INITIALIZED:
            raise Exception(f"method {self.classname()}.comm() called in a child process")
        else:
            raise Exception(f"method {self.classname()}.comm() called before initializing MPI")

    def local_comm(self):
        if self.mpi_state == MPIState.MPI_INITIALIZED:
            return self.mpi_local_comm
        elif self.mpi_state == MPIState.CHILD_INITIALIZED:
            raise Exception(f"method {self.classname()}.local_comm() called in a child process")
        else:
            raise Exception(f"method {self.classname()}.local_comm() called before initializing MPI")

    def local_rank(self):
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.size() called before initializing MPI")
        else:
            return self.mpi_local_rank

    def npernode(self):
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.size() called before initializing MPI")
        else:
            return self.mpi_ppn_list[self.mpi_node]
    def nnodes(self):
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.size() called before initializing MPI")
        else:
            return self.mpi_nodes
    
    def node(self):
        """
        Return the node index for this rank.
        """
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.node() called before initializing MPI")
        else:
            return self.mpi_node
    
    def reduce(self, num):
        from mpi4py import MPI
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.reduce() called before initializing MPI")
        else:
            return MPI.COMM_WORLD.allreduce(num, op=MPI.SUM)
    
    def finalize(self):
        from mpi4py import MPI
        if self.mpi_state == MPIState.MPI_INITIALIZED and MPI.Is_initialized():
            MPI.Finalize()

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        begin = time()
        x = func(*args, **kwargs)
        end = time()
        return x, "%10.10f" % begin, "%10.10f" % end, os.getpid()

    return wrapper


def progress(count, total, status=''):
    """
    Printing a progress bar. Will be in the stdout when debug mode is turned on
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + ">" + '-' * (bar_len - filled_len)
    if DLIOMPI.get_instance().rank() == 0:
        DLIOLogger.get_instance().info("\r[INFO] {} {}: [{}] {}% {} of {} ".format(utcnow(), status, bar, percents, count, total))
        if count == total:
            DLIOLogger.get_instance().info("")
        os.sys.stdout.flush()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def create_dur_event(name, cat, ts, dur, args={}):
    if "get_native_id" in dir(threading):
        tid = threading.get_native_id()
    elif "get_ident" in dir(threading):
        tid = threading.get_ident()
    else:
        tid = 0
    args["hostname"] = socket.gethostname()
    args["cpu_affinity"] = p.cpu_affinity()
    d = {
        "name": name,
        "cat": cat,
        "pid": DLIOMPI.get_instance().rank(),
        "tid": tid,
        "ts": ts * 1000000,
        "dur": dur * 1000000,
        "ph": "X",
        "args": args
    }
    return d

  
def get_trace_name(output_folder, use_pid=False):
    val = ""
    if use_pid:
        val = f"-{os.getpid()}"
    return f"{output_folder}/trace-{DLIOMPI.get_instance().rank()}-of-{DLIOMPI.get_instance().size()}{val}.pfw"
        
def sleep(config):
    sleep_time = 0.0
    if isinstance(config, dict) and len(config) > 0:
        if "type" in config:
            if config["type"] == "normal":
                sleep_time = np.random.normal(config["mean"], config["stdev"])
            elif config["type"] == "uniform":
                sleep_time = np.random.uniform(config["min"], config["max"])
            elif config["type"] == "gamma":
                sleep_time = np.random.gamma(config["shape"], config["scale"])
            elif config["type"] == "exponential":
                sleep_time = np.random.exponential(config["scale"])
            elif config["type"] == "poisson":
                sleep_time = np.random.poisson(config["lam"])
        else:
            if "mean" in config:
                if "stdev" in config:
                    sleep_time = np.random.normal(config["mean"], config["stdev"])
                else:
                    sleep_time = config["mean"]
    elif isinstance(config, (int, float)):
        sleep_time = config
    sleep_time = abs(sleep_time)
    if sleep_time > 0.0:
        base_sleep(sleep_time)
    return sleep_time


def _parse_csv_set(raw_value, cast=None):
    values = set()
    if not raw_value:
        return values
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        values.add(cast(token) if cast is not None else token)
    return values


def _fetch_delay_config():
    global _FETCH_DELAY_CFG
    if _FETCH_DELAY_CFG is not None:
        return _FETCH_DELAY_CFG
    delay_sec = float(os.environ.get("DLIO_INJECT_FETCH_DELAY_SEC", "0") or 0.0)
    _FETCH_DELAY_CFG = {
        "delay_sec": max(delay_sec, 0.0),
        "node_indexes": _parse_csv_set(
            os.environ.get("DLIO_INJECT_FETCH_DELAY_NODE_INDEXES", ""),
            cast=int,
        ),
        "global_ranks": _parse_csv_set(
            os.environ.get("DLIO_INJECT_FETCH_DELAY_GLOBAL_RANKS", ""),
            cast=int,
        ),
        "hosts": {host.lower() for host in _parse_csv_set(
            os.environ.get("DLIO_INJECT_FETCH_DELAY_HOSTS", ""),
        )},
    }
    return _FETCH_DELAY_CFG


def maybe_inject_fetch_delay(logger=None, *, location="fetch"):
    cfg = _fetch_delay_config()
    if cfg["delay_sec"] <= 0.0:
        return 0.0

    hostname = os.environ.get("DLIO_MPI_HOSTNAME", socket.gethostname())
    short_hostname = hostname.split(".", 1)[0].lower()

    target_by_index = bool(cfg["node_indexes"])
    target_by_rank = bool(cfg["global_ranks"])
    target_by_host = bool(cfg["hosts"])
    if not target_by_index and not target_by_rank and not target_by_host:
        matched = True
    else:
        matched = False
        node_index = os.environ.get("DLIO_MPI_NODE_INDEX", "")
        if target_by_index and node_index:
            try:
                matched = int(node_index) in cfg["node_indexes"]
            except ValueError:
                matched = False
        rank_label = os.environ.get(
            "DLIO_MPI_GLOBAL_RANK",
            os.environ.get("PMIX_RANK", os.environ.get("SLURM_PROCID", "")),
        )
        if not matched and target_by_rank and rank_label:
            try:
                matched = int(rank_label) in cfg["global_ranks"]
            except ValueError:
                matched = False
        if not matched and target_by_host:
            matched = (
                hostname.lower() in cfg["hosts"]
                or short_hostname in cfg["hosts"]
            )

    if not matched:
        return 0.0

    pid = os.getpid()
    if pid not in _FETCH_DELAY_LOGGED_PIDS:
        node_label = os.environ.get("DLIO_MPI_NODE_INDEX", "?")
        rank_label = os.environ.get(
            "DLIO_MPI_GLOBAL_RANK",
            os.environ.get("PMIX_RANK", os.environ.get("SLURM_PROCID", "?")),
        )
        marker = (
            f"{utcnow()} inject_fetch_delay active at {location}: "
            f"delay_sec={cfg['delay_sec']} node_index={node_label} "
            f"hostname={short_hostname} pid={pid} rank={rank_label}"
        )
        if logger is not None:
            logger.output(marker)
        try:
            sys.stderr.write(f"[DLIO_DELAY] {marker}\n")
            sys.stderr.flush()
        except Exception:
            pass
        try:
            output_folder = os.environ.get("DLIO_OUTPUT_FOLDER", "").strip()
            if output_folder:
                marker_dir = os.path.join(output_folder, "delay_markers")
                os.makedirs(marker_dir, exist_ok=True)
                marker_path = os.path.join(
                    marker_dir,
                    f"{short_hostname}-node{node_label}-rank{rank_label}-pid{pid}.log",
                )
                with open(marker_path, "a", encoding="utf-8") as marker_file:
                    marker_file.write(f"[DLIO_DELAY] {marker}\n")
        except Exception:
            pass
        _FETCH_DELAY_LOGGED_PIDS.add(pid)

    base_sleep(cfg["delay_sec"])
    return cfg["delay_sec"]

def gen_random_tensor(shape, dtype, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if not np.issubdtype(dtype, np.integer):
        # Only float32 and float64 are supported by rng.random
        if dtype not in (np.float32, np.float64):
            arr = rng.random(size=shape, dtype=np.float32)
            return arr.astype(dtype)
        else:
            return rng.random(size=shape, dtype=dtype)
    
    # For integer dtypes, generate float32 first then scale and cast
    dtype_info = np.iinfo(dtype)
    records = rng.random(size=shape, dtype=np.float32)
    records = records * (dtype_info.max - dtype_info.min) + dtype_info.min
    records = records.astype(dtype)
    return records
