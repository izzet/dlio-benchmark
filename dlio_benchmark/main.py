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
import math
import sys
from time import time, perf_counter
import numpy as np

# Reduce TF and CUDA logging

import hydra
from omegaconf import DictConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
# Remove PyTorch warning when libtorch_cuda_cu.so isn't found
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from dlio_benchmark.checkpointing.checkpointing_factory import CheckpointingFactory
from dlio_benchmark.common.constants import MODULE_DLIO_BENCHMARK
from dlio_benchmark.common.enumerations import DatasetType, MetadataType
from dlio_benchmark.utils.utility import utcnow, DLIOMPI, Profile, dft_ai, DLIOLogger
from dlio_benchmark.utils.statscounter import StatsCounter
from dlio_benchmark.utils.config import LoadConfig, ConfigArguments, GetConfig
from dlio_benchmark.profiler.profiler_factory import ProfilerFactory
from dlio_benchmark.framework.framework_factory import FrameworkFactory
from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
from dlio_benchmark.storage.storage_factory import StorageFactory

try:
    from dfoptimizer.runtime import tunable, knob, optimizer_context, Window
    _HAS_DFOPTIMIZER = True
except ImportError:
    _HAS_DFOPTIMIZER = False

dlp = Profile(MODULE_DLIO_BENCHMARK)
# To make sure the output folder is the same in all the nodes. We have to do this.

dftracer_initialize = True
dftracer_finalize   = True
dtracer             = None

class _StepDirect:
    """Direct monotonic account of the step body, rank 0 only.

    Section 4j of liveflow_shm/T3-0-SHM-LOOP-CORONA.md measured a 41.3 ms step at 32 nodes with
    only 0.46 ms of it waiting in `next()` and 9.36 ms of it sleeping: about 31 ms per step sits
    inside the step body and is attributed to nothing. FETCH_DIRECT answered where the wait is NOT.
    This answers where it is, with the same instrument (a monotonic clock, no trace region, no
    analyzer) applied to every other piece of the step in order, so the parts and the residual add
    up to the measured step.
    """

    __slots__ = ("_t", "acc", "steps", "total")

    def __init__(self):
        self._t = perf_counter()
        self.acc = {}
        self.steps = 0
        self.total = 0.0

    def begin(self):
        self._t = perf_counter()

    def mark(self, key):
        now = perf_counter()
        d = now - self._t
        self._t = now
        acc = self.acc
        acc[key] = acc.get(key, 0.0) + d
        return d

    def end_step(self, step_seconds):
        self.steps += 1
        self.total += step_seconds

    def report(self, epoch, rank, sleep_s):
        """One line per epoch, same shape as FETCH_DIRECT: sums, per-step ms, share of the step."""
        n = max(1, self.steps)
        parts = sorted(self.acc.items(), key=lambda kv: -kv[1])
        accounted = sum(self.acc.values())
        residual = self.total - accounted
        fields = " ".join(
            f"{k}={1000.0 * v / n:.3f}" for k, v in parts
        )
        return (
            f"STEP_DIRECT epoch={epoch} rank={rank} steps={self.steps} "
            f"step_ms={1000.0 * self.total / n:.3f} sleep_ms={1000.0 * sleep_s:.3f} "
            f"accounted_ms={1000.0 * accounted / n:.3f} "
            f"residual_ms={1000.0 * residual / n:.3f} | {fields}"
        )


class _NullStepDirect:
    """Every rank but 0 runs this: the calls stay in the code path, the cost is a no-op method."""

    __slots__ = ()

    def begin(self):
        pass

    def mark(self, key):
        return 0.0

    def end_step(self, step_seconds):
        pass


_NULL_STEP_DIRECT = _NullStepDirect()


class _AlreadyApplied(Exception):
    """This node already holds the requested data_source; do not resolve again."""


class DLIOBenchmark(object):
    """
    The Benchmark represents the I/O behavior of deep learning applications.
    """

    def __init__(self, cfg):
        """
        This initializes the DLIO benchmark. Intialization includes:
        <ul>
            <li> argument parser </li>
            <li> profiler instances </li>
            <li> internal components </li>
            <li> local variables </li>
        </ul>
        """
        global dftracer, dftracer_initialize, dftracer_finalize

        t0 = time()
        self.args = ConfigArguments.get_instance()
        LoadConfig(self.args, cfg)
        self.storage = StorageFactory().get_storage(self.args.storage_type, self.args.storage_root,
                                                    self.args.framework)

        self.output_folder = self.args.output_folder
        os.makedirs(self.args.output_folder, mode=0o755, exist_ok=True)
        os.environ["DLIO_OUTPUT_FOLDER"] = self.args.output_folder
        self.comm = DLIOMPI.get_instance().comm()
        self.local_comm = DLIOMPI.get_instance().local_comm()
        self.my_rank = self.args.my_rank = DLIOMPI.get_instance().rank()
        self.my_local_rank = self.args.my_local_rank = DLIOMPI.get_instance().local_rank()
        self.my_node = self.args.my_node = DLIOMPI.get_instance().node()
        self.comm_size = self.args.comm_size = DLIOMPI.get_instance().size()
        self.data_folder = self.args.data_folder
        self.storage_root = self.args.storage_root
        if self.args.storage_root:
            self.storage.create_namespace(exist_ok=True)
        self.framework = FrameworkFactory().get_framework(self.args.framework,
                                                          self.args.do_profiling)

        # Delete previous logfile
        if self.my_rank == 0:
            if os.path.isfile(self.args.logfile_path):
                os.remove(self.args.logfile_path)
        self.comm.barrier()
        # Configure the logging library
        self.args.configure_dlio_logging(is_child=False)
        self.logger = DLIOLogger.get_instance()
        if dftracer_initialize:
            dftracer = self.args.configure_dftracer(is_child=False, use_pid=False)
        with Profile(name=f"{self.__init__.__qualname__}", cat=MODULE_DLIO_BENCHMARK):
            mode = []
            if self.args.generate_data:
                mode += ["Generating data"]
            if self.args.do_train:
                mode += ["Training"]
            if self.args.do_eval:
                mode += ["Evaluation"]
            if self.args.do_checkpoint:
                mode += ["Checkpointing"]
            if self.args.my_rank == 0:
                self.logger.output(f"{utcnow()} Running DLIO [{' & '.join(mode)}] with {self.args.comm_size} process(es)")
                try:
                    self.logger.output(
                        f"{utcnow()} Reading workload YAML config file '{hydra_cfg.runtime.config_sources[1]['path']}/workload/{hydra_cfg.runtime.choices.workload}.yaml'")
                except:
                    pass
            self.generate_only = self.args.generate_only
            self.do_profiling = self.args.do_profiling

            self.data_generator = None
            self.num_files_train = self.args.num_files_train
            self.num_subfolders_train = self.args.num_subfolders_train
            self.num_subfolders_eval = self.args.num_subfolders_eval
            self.num_samples = self.args.num_samples_per_file
            self.total_training_steps = self.args.total_training_steps
            
            self.epochs = self.args.epochs
            self.batch_size = self.args.batch_size
            self.computation_time = self.args.computation_time

            if self.do_profiling:
                self.profiler = ProfilerFactory().get_profiler(self.args.profiler)

            if self.args.generate_data:
                self.data_generator = GeneratorFactory.get_generator(self.args.format)
            # Checkpointing support
            self.do_checkpoint = self.args.do_checkpoint
            self.steps_between_checkpoints = self.args.steps_between_checkpoints
            self.epochs_between_checkpoints = self.args.epochs_between_checkpoints
            self.checkpoint_after_epoch = self.args.checkpoint_after_epoch

            # Evaluation support
            self.do_eval = self.args.do_eval
            self.num_files_eval = self.args.num_files_eval

            self.batch_size_eval = self.args.batch_size_eval
            self.eval_time = self.args.eval_time
            self.eval_after_epoch = self.args.eval_after_epoch
            self.epochs_between_evals = self.args.epochs_between_evals
        self.stats = StatsCounter()

    def _emit_optimizer_debug(self, message):
        if os.environ.get("DFOPTIMIZER_DEBUG", "0") != "1":
            return
        if not (self.my_rank == 0 or self.my_local_rank == 0):
            return
        try:
            sys.stderr.write(
                f"{self.my_rank}: [DLIO_OPT_DEBUG] rank={self.my_rank} "
                f"local_rank={self.my_local_rank} node={self.my_node} "
                f"host={os.environ.get('DLIO_MPI_HOSTNAME', os.uname().nodename)} "
                f"{message}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass

    @dlp.log
    def initialize(self):
        """
        Initializes the benchmark runtime.
        - It generates the required data
        - Start profiling session for Darshan and Tensorboard.
        """
        self.comm.barrier()

        if self.args.generate_data:
            if self.args.my_rank == 0:
                self.logger.output(f"{utcnow()} Starting data generation")
            self.data_generator.generate()
            # important to have this barrier to ensure that the data generation is done for all the ranks
            self.comm.barrier()
            if self.args.my_rank == 0:
                self.logger.output(f"{utcnow()} Generation done")

        if not self.generate_only and self.do_profiling:
            self.profiler.start()
            self.framework.start_framework_profiler()
            self.comm.barrier()
            if self.args.my_rank == 0:
                self.logger.info(f"{utcnow()} Profiling Started with {self.args.profiler}")
        self.comm.barrier()
        file_list_train = []
        file_list_eval = []
        num_subfolders = 0
        if self.args.do_train:
            # Fast path: build file list synthetically when DLIO_SYNTHETIC_FILE_LIST=1.
            # Avoids Lustre metadata walk for millions of files (2M+ files can take >60 min).
            if os.environ.get("DLIO_SYNTHETIC_FILE_LIST", "0") == "1":
                fmt = self.args.format
                for dataset_type in [DatasetType.TRAIN, DatasetType.VALID]:
                    if dataset_type == DatasetType.TRAIN:
                        n = self.num_files_train
                    else:
                        n = self.num_files_eval if self.do_eval else 0
                    if n > 0:
                        base = os.path.join(self.args.data_folder, f"{dataset_type}")
                        pad = len(str(n - 1)) if n > 0 else 1
                        fullpaths = [
                            self.storage.get_uri(os.path.join(base, f"img_{i:0{pad}d}_of_{n}.{fmt}"))
                            for i in range(n)
                        ]
                        if dataset_type is DatasetType.TRAIN:
                            file_list_train = fullpaths
                        elif dataset_type is DatasetType.VALID:
                            file_list_eval = fullpaths
                if self.my_rank == 0:
                    self.logger.output(
                        f"{utcnow()} Synthetic file list: {len(file_list_train)} train, "
                        f"{len(file_list_eval)} eval (DLIO_SYNTHETIC_FILE_LIST=1)"
                    )
            else:
                for dataset_type in [DatasetType.TRAIN, DatasetType.VALID]:
                    if dataset_type == DatasetType.TRAIN:
                        num_subfolders = self.num_subfolders_train
                    else:
                        num_subfolders = self.num_subfolders_eval
                    filenames = self.storage.walk_node(os.path.join(self.args.data_folder, f"{dataset_type}"))
                    self.logger.debug(f"filenames {filenames} {num_subfolders}")
                    if (len(filenames) == 0):
                        continue
                    if self.storage.get_node(
                            os.path.join(self.args.data_folder, f"{dataset_type}",
                                        filenames[0])) == MetadataType.DIRECTORY:
                        assert (num_subfolders == len(filenames))
                        fullpaths = self.storage.walk_node(
                            os.path.join(self.args.data_folder, f"{dataset_type}/*/*.{self.args.format}"),
                            use_pattern=True)
                        files = [self.storage.get_basename(f) for f in fullpaths]
                        idx = np.argsort(files)
                        fullpaths = [fullpaths[i] for i in idx]
                        self.logger.debug(f"fullpaths {fullpaths}")
                    else:
                        assert (num_subfolders == 0)
                        fullpaths = [self.storage.get_uri(os.path.join(self.args.data_folder, f"{dataset_type}", entry))
                                    for entry in filenames if entry.endswith(f'{self.args.format}')]
                        fullpaths = sorted(fullpaths)
                        self.logger.debug(f"fullpaths {fullpaths}")
                    self.logger.debug(f"subfolder {num_subfolders} fullpaths {fullpaths}")
                    if dataset_type is DatasetType.TRAIN:
                        file_list_train = fullpaths
                    elif dataset_type is DatasetType.VALID:
                        file_list_eval = fullpaths
                if not self.generate_only and self.num_files_train > len(file_list_train):
                    raise Exception(
                        "Not enough training dataset is found; Please run the code with ++workload.workflow.generate_data=True")
                if self.do_eval and self.num_files_eval > len(file_list_eval):
                    raise Exception(
                        "Not enough evaluation dataset is found; Please run the code with ++workload.workflow.generate_data=True")
                if (self.num_files_train < len(file_list_train)):
                    self.logger.warning(
                        f"Number of files for training in {os.path.join(self.args.data_folder, f'{DatasetType.TRAIN}')} ({len(file_list_train)}) is more than requested ({self.num_files_train}). A subset of files will be used ")
                    file_list_train = file_list_train[:self.num_files_train]
            if (self.num_files_eval < len(file_list_eval)):
                self.logger.warning(
                    f"Number of files for evaluation in {os.path.join(self.args.data_folder, f'{DatasetType.VALID}')} ({len(file_list_eval)}) is more than requested ({self.num_files_eval}). A subset of files will be used ")
                file_list_eval = file_list_eval[:self.num_files_eval]
        self.args.derive_configurations(file_list_train, file_list_eval)
        self.args.validate()
        self.checkpointing_mechanism = None
        self.stats.checkpoint_size = 0
        if (not self.generate_only) and (self.do_checkpoint):
            self.checkpointing_mechanism = CheckpointingFactory().get_mechanism(self.args.checkpoint_mechanism)
            self.stats.checkpoint_size = self.checkpointing_mechanism.checkpoint_size
        self._optimizer_ctx = None
        self._optimizer_started = False
        self._cached_loader = None
        self._cached_prefetch = None
        self._cached_read_threads = None
        # Which epoch the cached loader was BUILT for. Reusing a loader across an epoch boundary is
        # unsound on the TF path: tf_data_loader.read() bakes epoch_number into the dataset (the
        # from_generator interleave at tf_data_loader.py:85,97) while next() increments it, so a
        # reused loader re-iterates a dataset wired to the PREVIOUS epoch's sample map with 8
        # interleaved Python generators. Observed at 32 nodes as a single rank of 256 either
        # segfaulting or wedging immediately after "Starting block 1", with zero steps completed --
        # once as a SIGSEGV entering epoch 2, once as a 1h44m hang entering epoch 3.
        self._cached_epoch = None
        self._ensure_optimizer_runtime_started()
        self.comm.barrier()

    def _ensure_optimizer_runtime_started(self):
        skip_reasons = []
        if self._optimizer_started:
            skip_reasons.append("already_started")
        if not _HAS_DFOPTIMIZER:
            skip_reasons.append("dfoptimizer_missing")
        if os.environ.get("DFOPTIMIZER_ENABLE", "0") != "1":
            skip_reasons.append("dfoptimizer_disabled")
        # WHO GETS AN OPTIMIZER CONTEXT.
        #
        # Restricting this to local rank 0 is CONTAINMENT, not design. Under Mofka every context
        # is a consumer living in the application's address space -- another client to fork, more
        # margo/argobots threads competing with NCCL (sec 57) -- so one per node was a way to
        # shrink the blast radius of a transport that could hang the rank holding it.
        #
        # It buys that at the cost of rank ASYMMETRY: only the holding rank ever learns a knob
        # changed, so the others keep stale values, and any apply path containing a collective
        # deadlocks on the ranks that never heard about it. That is sec 78, where a world
        # collective ran on 4 of 16 ranks.
        #
        # A broadcast mailbox removes the reason. A reader is an mmap and a version counter: no
        # connection, no threads, no progress engine, nothing to re-initialise across fork. Every
        # rank can afford one, so every rank gets one and symmetry stops being something the
        # actuator has to remember. Defaults to the old containment under Mofka.
        _per_rank = os.environ.get(
            "LIVEFLOW_CTX_PER_RANK",
            "1" if os.environ.get("LIVEFLOW_TRANSPORT", "mofka").strip().lower() != "mofka"
            else "0") == "1"
        if not _per_rank and self.my_local_rank != 0:
            skip_reasons.append("not_local_rank_zero")
        if skip_reasons:
            if self.my_rank == 0 or self.my_local_rank == 0:
                self._emit_optimizer_debug(
                    "optimizer_runtime_skipped "
                    f"reasons={','.join(skip_reasons)} "
                    f"group_file={os.environ.get('DFTRACER_MOFKA_GROUP_FILE', '')}"
                )
            return
        self._emit_optimizer_debug(
            "optimizer_runtime_start "
            f"group_file={os.environ.get('DFTRACER_MOFKA_GROUP_FILE', '')}"
        )

        group_file = os.environ.get("DFTRACER_MOFKA_GROUP_FILE", "")
        self._optimizer_ctx = optimizer_context(
            namespace="dlio",
            group_file=group_file,
            topic_plans="optimizer_plans",
            topic_acks="optimizer_acks",
            topic_registry="optimizer_registry",
        )
        self._optimizer_started = True

        try:
            tunable_func = getattr(self.make_loader, "__func__", self.make_loader)
            register = getattr(tunable_func, "_tunable_register", None)
            if callable(register):
                register(
                    self._optimizer_ctx,
                    current_values={
                        "prefetch_size": self.args.prefetch_size,
                        "read_threads": self.args.read_threads,
                    },
                )
                self._emit_optimizer_debug("optimizer_knobs_preregistered")
        except Exception as ex:
            self._emit_optimizer_debug(f"optimizer_knobs_preregister_failed error={ex}")
            self.logger.warning(f"{utcnow()} Failed to pre-register optimizer knobs: {ex}")

    @dft_ai.pipeline.evaluate
    def _eval(self, epoch):
        """
        Evaluation loop will read a separate dataset and has its own own computation time.
        """
        step = 1
        total = math.floor(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
        loader = self.eval_loader
        self.stats.start_loading()
        for batch in loader.next():
            # @ray: fixing uneven data fetch and computation count (same issue with `_train` below)
            # Check if max steps reached to prevent incomplete fetch/compute pairs
            # This ensures accurate event counting by stopping compute when step limit is hit
            if step > total:
                break
            self.stats.eval_batch_loaded(epoch, step)
            eval_time = self.eval_time
            self.stats.start_compute()
            self.framework.compute(batch, epoch, step, eval_time)
            self.stats.eval_batch_processed(epoch, step)
            step += 1
            self.stats.start_loading()
        return step - 1

    @dlp.log
    def _checkpoint(self):
        """
        Checkpointing loop will save the checkpoint after a certain number of steps.
        """
        self.stats.start_epoch()
        if self.args.num_checkpoints_write > 0:
            self._checkpoint_write()
        num_checkpoints_exists = len(self.storage.walk_node(self.args.checkpoint_folder))
        if num_checkpoints_exists < self.args.num_checkpoints_read:
            raise Exception("Number of checkpoints to be read: {self.args.num_checkpoints_read} is more than the number of checkpoints available: {num_checkpoints_exists}")
        if self.args.num_checkpoints_read > 0:
            self._checkpoint_read()
        self.stats.end_epoch()

    @dlp.log
    def _checkpoint_write(self):
        if self.comm.rank == 0:
            self.logger.output(f"{utcnow()} Checkpointing write started")
        block = 1  # A continuous period of training steps, ended by checkpointing
        block_step = overall_step = 1  # Steps are taken within blocks
        epoch = 1
        for i in dft_ai.checkpoint.iter(range(self.args.num_checkpoints_write), include_iter=False):
            dft_ai.checkpoint.start(metadata=True)
            # We still make sure that the checkpoint is done after allreduce; therefore, allreduce here is required.
            self.framework.compute(None, epoch, block_step, self.args.time_between_checkpoints)
            self.comm.barrier()
            self.stats.start_save_ckpt(epoch, block, overall_step)
            self.checkpointing_mechanism.save_checkpoint(epoch, overall_step)
            if self.args.checkpoint_rank_sync:
                self.comm.barrier()
            self.stats.end_save_ckpt(epoch, block)
            dft_ai.checkpoint.stop(metadata=True)
            block = block+1
            overall_step = overall_step + 1
        if self.comm.rank == 0:
            self.logger.output(f"{utcnow()} Checkpointing write finished")

    @dlp.log
    def _checkpoint_read(self):
        if self.comm.rank == 0:
            self.logger.output(f"{utcnow()} Checkpointing read started")
        block = 1  # A continuous period of training steps, ended by checkpointing
        block_step = overall_step = 1  # Steps are taken within blocks
        epoch = 1
        for i in dft_ai.checkpoint.iter(range(self.args.num_checkpoints_read), include_iter=False):
            dft_ai.checkpoint.start(metadata=True)
            self.framework.compute(None, epoch, block_step, self.args.time_between_checkpoints)
            self.comm.barrier()
            self.stats.start_load_ckpt(epoch, block, overall_step)
            self.checkpointing_mechanism.load_checkpoint(epoch, overall_step)
            if self.args.checkpoint_rank_sync:
                self.comm.barrier()
            self.stats.end_load_ckpt(epoch, block)
            dft_ai.checkpoint.stop(metadata=True)
            block = block+1
            overall_step = overall_step + 1
        if self.comm.rank == 0:
            self.logger.output(f"{utcnow()} Checkpointing write started")

    def _horizon_terms(self, epoch):
        """The terms an expensive actuator is judged against: steps, per-step compute, the epoch
        it should expect, and how many epochs are left after this one.

        WHY THIS IS NOT JUST `self._last_*`. Those three are set after `_train` returns, so at a
        WINDOW boundary they describe the PREVIOUS epoch or, in epoch 1, nothing at all -- and a
        zero epoch time gives gain 0, horizon 0, and an abort on the first chunk. That is why
        `data_source` was pinned to epoch boundaries. The pin then cost the 32-node run its result:
        the decision was ready 60 s after the epoch 1 boundary, waited for the epoch 2 boundary, and
        ran with a single epoch of horizon against an 884 s copy.

        Mid-epoch the same quantities are measurable rather than absent. Elapsed time and completed
        steps are both known, so the epoch this run is heading for is a projection from its own
        progress, not an estimate borrowed from elsewhere, and the remaining horizon includes the
        part of THIS epoch still to come. Before any step has completed there is still nothing to
        project from, and then the terms stay zero and an expensive action is declined, which is the
        same conservative answer as before.
        """
        steps_total = int(getattr(self, "_steps_per_epoch", 0) or 0)
        done = int(getattr(self, "_steps_done_this_epoch", 0) or 0)
        per_step = self._comp_seconds(self.args.computation_time)
        # Epochs strictly AFTER this one. `epoch` is one based.
        after = max(0, int(self.args.epochs) - int(epoch))

        if done > 0 and steps_total > 0:
            elapsed = max(1e-6, time() - getattr(self, "_epoch_t0", time()))
            frac_done = min(1.0, done / float(steps_total))
            projected = elapsed / max(frac_done, 1e-6)
            # The rest of this epoch counts toward the payback too: staging now shortens the
            # remainder of the epoch it interrupts, not only the ones after it.
            remaining = after + (1.0 - frac_done)
            return {"steps": steps_total, "per_step_compute": per_step,
                    "epoch_seconds": projected, "epochs_remaining": remaining}

        # Epoch boundary, or a window boundary before the first step completed: fall back to the
        # epoch that actually finished, which is exact when it exists.
        return {"steps": getattr(self, "_last_steps", 0),
                "per_step_compute": getattr(self, "_last_per_step_compute", per_step),
                "epoch_seconds": getattr(self, "_last_epoch_seconds", 0.0),
                "epochs_remaining": after + 1}

    def make_loader(self, epoch, prefetch_size=None, read_threads=None, data_source=None):
        """Create or reuse a training DataLoader for the given epoch.

        When prefetch_size and read_threads are unchanged from the previous
        epoch, the cached DataLoader (with its persistent workers) is reused
        to avoid per-epoch fork/init overhead.  A new DataLoader is created
        only when knob values actually change.

        Each node's local-rank-0 process receives overrides from the
        @tunable decorator; values are then broadcast within the node so
        every local worker uses the same settings.
        """
        if prefetch_size is None:
            prefetch_size = self.args.prefetch_size
        if read_threads is None:
            read_threads = self.args.read_threads
        # data_source arrives like every other knob -- injected by @tunable from a plan off the SHM
        # mailbox. What satisfying a value COSTS is not the application's business: resolvers are
        # registered elsewhere, so adding a data-location actuator ("replica", "compressed",
        # "cache", ...) touches no code here. A resolver that declines or fails returns None and the
        # run carries on unchanged -- an actuator must never take down the job it is optimising.
        #
        # THE AGREEMENT IS COLLECTIVE, IT IS TAKEN OUTSIDE THE CHANGE GUARD, AND IT IS NODE-LOCAL.
        #
        # `data_source` is the one knob whose apply contains a collective (capacity veto, chunk
        # count, abort ballot). The mailbox that carries the plan is reconciling, not
        # synchronising: desired_actions_for's own docstring says every rank converges on the same
        # value EVENTUALLY, not at the same instant, and that this "is NOT sufficient for a knob
        # whose apply contains a collective ... a rank arriving a window late deadlocks the ones
        # that arrived on time". Every rank reaches this line at every epoch boundary whether or
        # not it saw a plan, so the decision is made collective too: agree first, then act.
        #
        # NODE-LOCAL, and this is not a weakening -- it is the scope the architecture already has.
        # Measured on the first two-node run: node 1's agent proposed data_source=local and all
        # eight of its ranks saw it, node 0's agent had not, and a WORLD allreduce(min) therefore
        # deferred forever. That is structural, not a race. Agents are per node with no channel
        # between them (that independence is the design, verified at 2 nodes in
        # STATUS-CORONA-2026-09-01 section 1.6), so a world-unanimous vote can only succeed if two
        # independent planners happen to decide the same thing in the same epoch.
        #
        # And staging IS a per-node action: each node copies its own shard to its own NVMe. The
        # global file LIST stays identical either way -- a staged node holds real files, an
        # unstaged node holds the PFS paths, both lists have the same names in the same order, and
        # tf_reader/dlio_sampler shard by INDEX -- so a node that declines simply reads from the
        # PFS. Mixed is safe here; what would not be safe is two ranks OF ONE NODE disagreeing,
        # because they share the destination directory, and that is exactly what this vote covers.
        #
        # It also removes a real availability problem: free space on /l/ssd was measured between
        # 1 GB and 1.7 TB across nodes of this cluster on one day, and under a world veto one full
        # disk blocked staging everywhere.
        _want = 1 if (data_source and data_source != getattr(self, "_data_source", "pfs")) else 0
        _agreed = _want
        _vote_comm = self.local_comm if self.local_comm is not None else self.comm
        if _vote_comm is not None:
            try:
                # op=min, not MPI.MIN: mpi4py maps the builtin, and main.py deliberately does not
                # import mpi4py at module scope (fork-safety, sec 57). staging_action.py uses the
                # same form for its own veto.
                _agreed = _vote_comm.allreduce(_want, op=min)
            except Exception as e:                      # noqa: BLE001 - never fail the training run
                self.logger.output(f"{utcnow()} data_source agreement unavailable: {e}")
                _agreed = 0
        if _want and not _agreed:
            self.logger.output(
                f"{utcnow()} data_source={data_source} deferred: not every rank sees it yet "
                f"(reconciling delivery; will re-assert at the next epoch boundary)")
        if _agreed:
            try:
                import sys as _sys
                _sys.path.insert(0, os.environ.get(
                    "LIVEFLOW_PY",
                    "/p/vast1/izzet/worktrees/dfsuite/corona/liveflow_shm/python"))
                from data_source import resolve as _resolve
                # node/nnodes/ppn come from MPI, NOT from an environment variable. The resolver used
                # to read LIVEFLOW_NODES with a default of 1, which no harness sets: ppn then came
                # out as comm_size, every rank computed node=0, and every node staged node 0's
                # shard. That is PAPER-SPINE integration bug 4 through a different door, and it does
                # not fail -- it stages 1/N of the corpus N times and reads the rest over symlinks,
                # so the arm reports a clean, believable "staging barely helps".
                # `comm` is the NODE-LOCAL communicator: the resolver's collectives (capacity veto,
                # chunk-count agreement, abort ballot) are about one node's own copy to its own
                # disk, and the agreement above was taken on the same communicator. `comm_size`
                # stays the WORLD size because the shard mapping is a function of global rank
                # count, not of this node's.
                _ctx = type("Ctx", (), {
                    "comm": (self.local_comm if self.local_comm is not None else self.comm),
                    "rank": self.my_rank, "local_rank": self.my_local_rank,
                    "comm_size": self.comm_size, "framework": self.args.framework,
                    "node": self.my_node, "nnodes": DLIOMPI.get_instance().nnodes(),
                    "ppn": DLIOMPI.get_instance().npernode(),
                    **self._horizon_terms(epoch),
                    "log": self.logger.output})()
                if self.my_rank == 0:
                    self.logger.output(
                        f"{utcnow()} data_source -> {data_source}: horizon from epoch "
                        f"{_ctx.epoch_seconds:.2f}s, {_ctx.steps} steps x "
                        f"{_ctx.per_step_compute:.5f}s compute, {_ctx.epochs_remaining} epochs "
                        f"remaining, {_ctx.nnodes} nodes x {_ctx.ppn} ranks")
                # IDEMPOTENT BY CONSTRUCTION. A shard already committed on this node is not
                # copied again, whatever asks: the resolver is the expensive path and the guard
                # belongs in front of it, not inside it. Without this the only thing standing
                # between a repeated plan and a repeated 500 s copy was a state assignment that an
                # exception could skip, which is exactly what happened (4s.32).
                if data_source == self._data_source:
                    self.logger.output(
                        f"{utcnow()} data_source ack: {self._data_source} -> {data_source} "
                        f"reason=no_move_needed applied=False completed_at=None "
                        f"files={len(self.args.file_list_train)} "
                        f"data_folder={self.args.data_folder}")
                    self._data_source_reason = "no_move_needed"
                    self._data_source_applied = False
                    raise _AlreadyApplied()
                _new = _resolve(data_source, list(self.args.file_list_train), _ctx)
                # THE ACTUATOR'S OWN REASON, RECORDED (T3-0 4s.30).
                #
                # `cf_fz_loop_n32_r1b` published `data_source local -> pfs` on all 32 agents, every
                # ack came back `applied`, and NOTHING MOVED: the run kept the staged speed and the
                # shards were released only at exit. A revert that changes nothing was
                # indistinguishable in the record from one that was carried out, which is not a
                # state the paper can describe. The resolver returning no file list is the actual
                # outcome, so it is said out loud and stamped on the knob's own reported state.
                _reason = ("relocated" if _new else
                           ("no_move_needed" if data_source == self._data_source
                            else "resolver_declined"))
                # WHEN THE COPY FINISHED, which only this component knows (T3-0 4s.31). The
                # resolver is synchronous, so the moment it returns the relocation is complete and
                # `_t_reloc` is the wall it took. The planner needs this because the copy's own
                # windows are neither the old regime nor the new one: sampling the moved arm from
                # the plan instead of from here is what reverted a decision that beat its own
                # prediction by 2.7x (4s.30).
                # `time` IS THE FUNCTION HERE, not the module: line 21 rebinds it with
                # `from time import time`. Writing `time.time()` raised
                # `'builtin_function_or_method' object has no attribute 'time'` on every
                # application, and because it raised AFTER the copy had already run and BEFORE the
                # state below was assigned, the shard was copied and then forgotten: DLIO kept the
                # PFS file list and the next make_loader resolved again. That is the 160 copies of
                # `cf_fz_loop_n32_r1c`, five per node, and no epoch ever finished (T3-0 4s.32).
                _completed_s = time() if _reason == "relocated" else None
                # STATE FIRST, LOGGING AFTER. The copy is the expensive, irreversible part; once it
                # has happened nothing later in this block may be allowed to lose the fact.
                if _new:
                    self.args.file_list_train = _new
                    self.args.data_folder = os.path.dirname(os.path.dirname(_new[0]))
                self._data_source = data_source
                self._data_source_reason = _reason
                self._data_source_applied = (_reason == "relocated")
                self._data_source_completed_at = _completed_s
                self.logger.output(
                    f"{utcnow()} data_source ack: {self._data_source} -> {data_source} "
                    f"reason={_reason} applied={_reason == 'relocated'} "
                    f"completed_at={_completed_s} files={len(_new) if _new else 0} "
                    f"data_folder={self.args.data_folder}")
            except _AlreadyApplied:
                pass
            except Exception as e:
                self.logger.output(f"{utcnow()} data_source resolve unavailable: {e}")

        # Broadcast knob values from local rank 0 so per-node optimizers can
        # tune different nodes independently.
        knobs = [prefetch_size, read_threads]
        knobs = self.local_comm.bcast(knobs, root=0)
        prefetch_size, read_threads = knobs

        self.args.prefetch_size = prefetch_size
        self.args.read_threads = read_threads

        # Reuse cached loader only WITHIN an epoch, and only if knobs haven't changed. Crossing an
        # epoch boundary must rebuild (see _cached_epoch above); line 669's comment already states
        # the intent -- "Create fresh train loader each epoch" -- which the cache was defeating.
        same_epoch = self._cached_epoch == epoch
        if (
            self._cached_loader is not None
            and same_epoch
            and self._cached_prefetch == prefetch_size
            and self._cached_read_threads == read_threads
        ):
            if self.my_rank == 0:
                self.logger.output(
                    f"{utcnow()} make_loader: epoch={epoch} prefetch={prefetch_size}"
                    f" threads={read_threads} (cached)"
                )
            loader = self._cached_loader
        elif (
            self._cached_loader is not None
            and same_epoch
            and hasattr(self._cached_loader, 'reconfigure')
        ):
            # LiveDataLoader: reconfigure in-place, return same object so
            # the active batch_iter generator in _train stays alive.
            self._cached_loader.reconfigure(
                read_threads=read_threads, prefetch_size=prefetch_size)
            self._cached_prefetch = prefetch_size
            self._cached_read_threads = read_threads
            self._cached_epoch = epoch
            if self.my_rank == 0:
                self.logger.output(
                    f"{utcnow()} make_loader: epoch={epoch} prefetch={prefetch_size}"
                    f" threads={read_threads} (reconfigure)"
                )
            loader = self._cached_loader
        else:
            # A persistent PyTorch DataLoader keeps its worker processes after an epoch.  Do not
            # fork this epoch's pool until the previous one has exited.  With the LiveFlow gate a
            # rank's pool is deliberately wide, so overlapping pools transiently doubled the
            # worker count and fork-inherited tracer/gate state at every epoch boundary.
            previous = self._cached_loader
            retire = getattr(previous, "shutdown_workers", None)
            if callable(retire):
                retire("epoch %s -> %s" % (self._cached_epoch, epoch))
            if self.my_rank == 0:
                self.logger.output(
                    f"{utcnow()} make_loader: epoch={epoch} prefetch={prefetch_size}"
                    f" threads={read_threads} (new)"
                )
            loader = self.framework.get_loader(dataset_type=DatasetType.TRAIN, epoch=epoch)
            loader.read(read_threads=read_threads, prefetch_size=prefetch_size)
            self._cached_loader = loader
            self._cached_prefetch = prefetch_size
            self._cached_read_threads = read_threads
            self._cached_epoch = epoch

        self._ensure_optimizer_runtime_started()

        return loader

    @staticmethod
    def _comp_seconds(ct):
        """computation_time may be a scalar OR a distribution dict.

        DLIO accepts e.g. {'mean': 0.00936, 'stdev': ...} and its own log prints
        "(set value: {'mean': 0.00936})". Multiplying that by a step count raised
        TypeError: unsupported operand type(s) for *: 'dict' and 'int' -- which killed the first
        live-controller run AFTER epoch 1 had already completed, so the failure looked like a
        crash rather than a type error in the hook.
        """
        if isinstance(ct, dict):
            for k in ("mean", "value", "time"):
                if k in ct:
                    return float(ct[k])
            return float(next(iter(ct.values()), 0.0))
        return float(ct)

    def _window_will_close(self):
        """Whether this step's stop call will publish the active window."""
        window = self._window
        if window is None or not getattr(window, "_active", False):
            return False
        return window.every_n <= 1 or window.counter % window.every_n == 0

    @staticmethod
    def _emit_loader_stall(loader, epoch, step):
        """Drain the loader account before window.stop reaches the analyzer."""
        emit = getattr(loader, "emit_window_stall", None)
        if callable(emit):
            return emit(epoch, step)
        return 0.0


    @dft_ai.pipeline.train
    def _train(self, epoch, loader):
        """
        Training loop for reading the dataset and performing training computations.
        :return: returns total steps.
        """
        block = 1  # A continuous period of training steps, ended by checkpointing
        block_step = overall_step = 1  # Steps are taken within blocks
        max_steps = math.floor(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
        self.steps_per_epoch = max_steps
        # Start the very first block
        self.stats.start_block(epoch, block)
        self.stats.start_loading()
        batch_iter = loader.next()
        # See _StepDirect: the direct account of every piece of the step body, rank 0 only, so the
        # ~31 ms/step that section 4j found outside next() and outside the sleep is attributed to a
        # named call rather than to "the step body".
        direct = self._step_direct = (
            _StepDirect() if self.my_rank == 0 else _NULL_STEP_DIRECT
        )
        while True:
            _step_t0 = perf_counter()
            direct.begin()
            dft_ai.update(epoch=epoch, step=overall_step, args={"max_steps": max_steps})
            # Window boundary (gated by every_n) — replaces pipeline.step
            # control events. Only emits to Mofka at cadence boundaries.
            if self._window is not None:
                self._window.start()
            direct.mark("update_window_start")   # dft_ai.update + window.start
            dft_ai.pipeline.step.start(metadata=True)
            direct.mark("step_start")
            try:
                batch = next(batch_iter)
            except StopIteration:
                dft_ai.pipeline.step.stop(metadata=True)
                if self._window is not None:
                    if self._window_will_close():
                        self._emit_loader_stall(loader, epoch, overall_step)
                    self._window.stop()
                break
            # Check if max steps reached to prevent incomplete fetch/compute pairs
            if overall_step > max_steps or ((self.total_training_steps > 0) and (overall_step > self.total_training_steps)):
                if self.args.my_rank == 0:
                    self.logger.info(f"{utcnow()} Maximum number of steps reached")
                if (block_step != 1 and self.do_checkpoint) or (not self.do_checkpoint):
                    self.stats.end_block(epoch, block, block_step - 1)
                dft_ai.pipeline.step.stop(metadata=True)
                if self._window is not None:
                    if self._window_will_close():
                        self._emit_loader_stall(loader, epoch, overall_step)
                    self._window.stop()
                break
            direct.mark("next")
            self.stats.batch_loaded(epoch, overall_step, block)
            direct.mark("stats_batch_loaded")
            # PARENT compute region, named `compute`, per liveflow_shm/ANNOTATION-SPEC.md.
            #
            # Was dft_ai.compute.step. The rename is numerically neutral -- same wrapped code, same
            # span, and Compute.__init__ sets cat=COMPUTE name=COMPUTE -- but it is what makes the
            # domain time selectable as the PARENT of a hierarchy. A layer definition of
            # `cat == "compute"` sums every level, so the moment a child region is added under a
            # broad definition, compute silently reports ~2x. Selecting `func_name == "compute"`
            # cannot double count at any depth; sub-regions (forward/backward) can then be added
            # freely as diagnostics, which is what StorMer already does.
            # DURATIONAL, not metadata. The analyzer's domain split assumes the compute region is
            # the PARENT that contains communication: metrics.py computes
            # `compute_time = compute_total - comm` (tests/test_metrics.py says so in as many
            # words). Until 2026-09-03 the only durational region named `compute` was
            # TFFramework.compute, i.e. the sleep alone, which does NOT contain the barrier below,
            # so booking the barrier under comm would have subtracted it from the sleep and driven
            # the compute domain to 0. Emitting the parent here restores the hierarchy the analyzer
            # documents. The frameworks' own region is now the child `compute.step`, so exactly one
            # region is named `compute` and the preset's narrow definition cannot double count.
            dft_ai.compute.start()
            direct.mark("compute_annot_start")
            try:
                computation_time = self.args.computation_time
                if (isinstance(computation_time, dict) and len(computation_time) > 0) or (isinstance(computation_time, float) and  computation_time > 0):
                    self.framework.trace_object("Train", overall_step, 1)
                self.stats.start_compute()
                direct.mark("stats_start_compute")
                # framework.compute() is the sleep AND, on the TensorFlow path, whatever the
                # framework does with the batch. If next() returned a lazy batch, the read
                # materialises in here, and fw_compute comes out far above sleep_ms.
                self.framework.compute(batch, epoch, block_step, self.computation_time)
                direct.mark("fw_compute")
                self.stats.batch_processed(epoch, overall_step, block)
                direct.mark("stats_batch_processed")
                # This is the barrier to simulate allreduce. It is required to simulate the actual workloads.
                #
                # ANNOTATED, because at scale it is the largest term in the step and it used to
                # belong to no domain. Section 4k of liveflow_shm/T3-0-SHM-LOOP-CORONA.md measured
                # 29.671 ms here against a 40.143 ms step at 32 nodes, 73.9% of the step, and the
                # analyzer reported it as `unattributed` (comm was 0.000 s in every window) because
                # nothing emitted a comm region. Rank 0 is not waiting on its own storage; it is
                # waiting for whichever of the other 255 ranks is, so this is where a
                # bulk-synchronous step re-expresses the slow rank's input time. `all_reduce` is
                # the child name DLIO's own comment already claims and the one StorMer and DLRM
                # emit, and the comm layer definition is broad, so children need no parent.
                #
                # Rank-symmetric by construction: every rank runs this line, so no rank enters or
                # leaves the region that does not also enter or leave the barrier.
                with dft_ai.comm.all_reduce:
                    self.comm.barrier()
                direct.mark("comm_barrier")
                if self.do_checkpoint and (
                        self.steps_between_checkpoints >= 0) and overall_step == self.next_checkpoint_step:
                    self.stats.end_block(epoch, block, block_step)
                    self.stats.start_save_ckpt(epoch, block, overall_step)
                    self.checkpointing_mechanism.save_checkpoint(epoch, overall_step)
                    self.stats.end_save_ckpt(epoch, block)
                    block += 1
                    # Reset the number of steps after every checkpoint to mark the start of a new block
                    block_step = 1
                    self.next_checkpoint_step += self.steps_between_checkpoints
                else:
                    block_step += 1
            finally:
                dft_ai.compute.stop()
                direct.mark("compute_annot_stop")
            dft_ai.pipeline.step.stop(metadata=True)
            direct.mark("step_stop")
            # LIVE EPOCH PROGRESS, so a horizon can be priced mid-epoch instead of only after the
            # epoch has finished. See section 4e of liveflow_shm/T3-0-SHM-LOOP-CORONA.md: at 32
            # nodes the cost-ranked staging decision was ready 60 s AFTER the epoch 1 boundary, so
            # under an epoch-boundary-only rule it waited for the epoch 2 boundary and ran with one
            # epoch of horizon, which correctly aborted. Nothing was wrong with the decision or the
            # abort; the actuator was simply offered the wrong moment.
            self._steps_done_this_epoch = overall_step
            if self._window is not None:
                # The accumulated event must be ordered before window.stop in the ring. The
                # analyzer drains on window.stop, so emitting it afterwards silently assigns the
                # storage wait to the next window or loses the final one at epoch end.
                if self._window_will_close():
                    self._emit_loader_stall(loader, epoch, overall_step)
                at_boundary = self._window.stop()
                direct.mark("window_stop")
                # Mid-epoch plan application: at window boundaries, check if
                # the optimizer has published new knobs. If so, rebuild the
                # loader with updated parallelism (e.g., read_threads).
                if at_boundary:
                    new_loader = self.make_loader(
                        epoch=epoch,
                        prefetch_size=self.args.prefetch_size,
                        read_threads=self.args.read_threads,
                        # @tunable only injects a knob whose parameter is PRESENT in kwargs
                        # (tunable.py:110); anything else is acked "rejected". data_source was
                        # declared on make_loader but never passed here, so every data_source plan
                        # was rejected with knob_not_in_kwargs -- 7 times in one run -- and the
                        # actuator could never fire however cheap or correct the plan was.
                        data_source=getattr(self, "_data_source", "pfs"),
                        _apply_at="window_boundary",
                    )
                    if new_loader is not loader:
                        loader.finalize()
                        loader = new_loader
                        batch_iter = loader.next()
                    # make_loader is where the gate and optimizer_context calls live
                    # (_ensure_optimizer_runtime_started), so this covers both.
                    direct.mark("boundary_rebuild")
            overall_step += 1
            # start a new block here
            if block_step == 1 and block != 1:
                self.stats.start_block(epoch, block)
            self.stats.start_loading()
            direct.mark("stats_start_loading")
            direct.end_step(perf_counter() - _step_t0)

        self.comm.barrier()
        if self.do_checkpoint and (self.steps_between_checkpoints < 0) and (epoch == self.next_checkpoint_epoch):
            self.stats.end_block(epoch, block, block_step-1)
            self.stats.start_save_ckpt(epoch, block, overall_step-1)
            self.checkpointing_mechanism.save_checkpoint(epoch, overall_step)
            self.stats.end_save_ckpt(epoch, block)
            self.next_checkpoint_epoch += self.epochs_between_checkpoints
        return overall_step

    @dft_ai
    def run(self):
        """
        Run the total epochs for training. 
        On each epoch, it prepares dataset for reading, it trains, and finalizes the dataset.
        If evaluation is enabled, it reads the eval dataset, performs evaluation and finalizes.
        """
        self.stats.start_run()
        if (not self.generate_only) and (not self.args.checkpoint_only):
            # Print out the expected number of steps for each epoch and evaluation
            if self.my_rank == 0:
                total = math.floor(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
                self.logger.output(
                    f"{utcnow()} Max steps per epoch: {total} = {self.num_samples} * {self.num_files_train} / {self.batch_size} / {self.comm_size} (samples per file * num files / batch size / comm size)")
                if self.total_training_steps > 0:
                    self.logger.output(
                        f"{utcnow()} Total training steps is set to be {self.total_training_steps}. Will only run up to {min(total*self.args.epochs, self.total_training_steps)}"
                    )
                if self.do_eval:
                    total = math.floor(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
                    self.logger.output(
                        f"{utcnow()} Steps per eval: {total} = {self.num_samples} * {self.num_files_eval} / {self.batch_size_eval} / {self.comm_size} (samples per file * num files / batch size eval / comm size)")

            # Keep track of the next epoch at which we will evaluate
            next_eval_epoch = self.eval_after_epoch
            self.next_checkpoint_epoch = self.checkpoint_after_epoch
            epoch = 1
            # Initialize the dataset (stores format/loader type, creates storage)
            self.args.reconfigure(epoch)
            self.framework.init_loader(self.args.format, epoch=epoch, data_loader=self.args.data_loader)
            # Prepare eval loader once (not tunable)
            if self.do_eval:
                self.eval_loader = self.framework.get_loader(dataset_type=DatasetType.VALID, epoch=epoch)
                self.eval_loader.read()
            # Create analysis window with self-tuning cadence.
            # Default: ~5 windows per epoch — enough for mid-epoch tuning
            # without thrashing the pipeline on high-step workloads (e.g.,
            # CosmoFlow has 98K steps/epoch; old min(1000, N) gave 98 windows).
            # DLIO_WINDOW_EVERY_N env var overrides this default.
            _steps_per_epoch = math.floor(
                self.num_samples * self.num_files_train / self.batch_size / self.comm_size
            )
            # Kept on the object so a mid-epoch horizon can be projected from progress.
            self._steps_per_epoch = _steps_per_epoch
            # DERIVED FROM THE WORKLOAD, never passed in. See liveflow_shm/HORIZON-CONTRACT.md.
            #
            # A cadence carried over from another cell is a silent way to switch the controller off.
            # `every_n=512` was chosen for cosmoflow, which runs 12288 steps per epoch, and then
            # reused for unet3d, which runs 108: the window could never close on cadence, only the
            # epoch-end flush closed it, and the whole 3-epoch run produced 5 windows per agent with
            # a median wall of 186 s, one entire epoch. With `min_samples=3` nothing could be
            # characterised even in principle (T3-0 section 4o.2).
            #
            # steps_per_epoch // 10 gives ~10 windows per epoch on any workload, which is the
            # quantity that actually matters: enough windows for the sample floor to be reachable,
            # few enough that the analyzer keeps up.
            _default_every_n = max(1, _steps_per_epoch // 10)
            _initial_every_n = int(os.environ.get(
                "DLIO_WINDOW_EVERY_N", _default_every_n
            ))
            if _HAS_DFOPTIMIZER and dftracer:
                # epochs_total and steps_per_epoch are the two facts only the application knows,
                # and they are what the horizon contract (liveflow_shm/HORIZON-CONTRACT.md) needs:
                # the window carries them onto every boundary event so the analyzer can publish
                # epochs_remaining and windows_remaining, and a cost-aware planner can price a
                # 465 s stage against the run that is actually left instead of declining because
                # it cannot see how much run there is.
                self._window = Window(
                    dftracer.get_instance(),
                    max_every_n=_steps_per_epoch,
                    epochs_total=self.epochs,
                    steps_per_epoch=_steps_per_epoch,
                )
                self._window.update_cadence(_initial_every_n)
            else:
                self._window = None

            self.comm.barrier()
            for epoch in dft_ai.pipeline.epoch.iter(range(1, self.epochs + 1), include_iter=False):
                dft_ai.pipeline.epoch.start(metadata=True)
                self.stats.start_epoch(epoch)
                # Horizon contract: the window stamps epoch_index on every boundary event from
                # here on. Set BEFORE make_loader, so the boundary that carries a staging decision
                # already knows which epoch it is in.
                if self._window is not None:
                    self._window.set_epoch(epoch)
                self._epoch_t0 = time()   # for the live-actuation hook's epoch_seconds
                self._steps_done_this_epoch = 0
                self.next_checkpoint_step = self.steps_between_checkpoints
                # Create fresh train loader each epoch (subject to @tunable overrides)
                loader = self.make_loader(
                    epoch=epoch,
                    prefetch_size=self.args.prefetch_size,
                    read_threads=self.args.read_threads,
                    # present in kwargs or @tunable cannot inject it; see the window-boundary call
                    data_source=getattr(self, "_data_source", "pfs"),
                )
                self.stats.start_train(epoch)
                steps = self._train(epoch, loader)
                self.stats.end_train(epoch, steps)
                # Record what the reconcile will need if a plan arrives. No policy here: the decision
                # is the optimizer's and reaches us as a `data_source` knob value off the mailbox.
                # T1.2 lead B: the DIRECT account of the consumer boundary, measured with a
                # monotonic clock around exactly `next(source_iterator)` and independent of every
                # trace region. Printed per epoch on rank 0 so the trace's fetch.iter sum can be
                # compared against it for the same rank: if they disagree, the region is not where
                # the wait is, and no amount of analyzer work will find the storage time.
                try:
                    _cb = getattr(loader, "_consumer_boundary", None)
                    if _cb is not None and self.my_rank == 0:
                        _stall_s, _stall_steps = _cb.totals
                        _ep_s = max(1e-9, time() - getattr(self, "_epoch_t0", time()))
                        self.logger.output(
                            f"{utcnow()} FETCH_DIRECT epoch={epoch} rank=0 "
                            f"next_wait_sum_s={_stall_s:.3f} steps={_stall_steps} "
                            f"per_step_ms={1000.0 * _stall_s / max(1, _stall_steps):.3f} "
                            f"epoch_s={_ep_s:.3f} frac_of_epoch={_stall_s / _ep_s:.3f}")
                except Exception as _e:                    # noqa: BLE001 - never fail the run
                    self.logger.output(f"{utcnow()} FETCH_DIRECT unavailable: {_e}")
                # The rest of the step body, same instrument, same rank, same epoch. Read it
                # against FETCH_DIRECT: next() plus the named parts plus the residual is the step.
                try:
                    _sd = getattr(self, "_step_direct", None)
                    if _sd is not None and self.my_rank == 0 and getattr(_sd, "steps", 0):
                        self.logger.output(
                            f"{utcnow()} "
                            + _sd.report(epoch, 0, self._comp_seconds(self.args.computation_time))
                        )
                except Exception as _e:                    # noqa: BLE001 - never fail the run
                    self.logger.output(f"{utcnow()} STEP_DIRECT unavailable: {_e}")
                self._last_steps = steps
                self._last_per_step_compute = self._comp_seconds(self.args.computation_time)
                self._last_epoch_seconds = max(1e-6, time() - getattr(self, "_epoch_t0", time()))
                # Force window boundary at epoch end so analyzer processes
                # any accumulated events before the next epoch begins.
                if self._window is not None:
                    if getattr(self._window, "_active", False):
                        self._emit_loader_stall(loader, epoch, steps)
                    self._window.flush()
                self.logger.debug(f"{utcnow()} Rank {self.my_rank} returned after {steps} steps.")
                loader.finalize()
                # Perform evaluation if enabled
                if self.do_eval and epoch >= next_eval_epoch:
                    next_eval_epoch += self.epochs_between_evals
                    self.stats.start_eval(epoch)
                    self._eval(epoch)
                    self.stats.end_eval(epoch)
                self.args.reconfigure(epoch + 1) # reconfigure once per epoch
                self.stats.end_epoch(epoch)
                dft_ai.pipeline.epoch.stop(metadata=True)
            if self.do_eval:
                self.eval_loader.finalize()

        if (self.args.checkpoint_only):
            self._checkpoint()            
        self.stats.end_run()

    @dlp.log
    def finalize(self):
        """
        It finalizes the dataset once training is completed.
        """

        global dftracer, dftracer_initialize, dftracer_finalize

        self.comm.barrier()
        if self.checkpointing_mechanism:
            self.checkpointing_mechanism.finalize()
        if not self.generate_only:
            if self.do_profiling:
                self.profiler.stop()
                self.framework.stop_framework_profiler()
                self.comm.barrier()
                if self.my_rank == 0:
                    self.logger.info(f"{utcnow()} Profiling stopped")
            if not self.args.keep_files:
                self.logger.info(f"{utcnow()} Keep files set to False. Deleting dataset")
                self.comm.barrier()
                if self.my_rank == 0:
                    if self.storage.get_node(self.args.data_folder):
                        self.storage.delete_node(self.args.data_folder)
                        self.logger.info(f"{utcnow()} Deleted data files")

            # Save collected stats to disk
            self.stats.finalize()
            self.stats.save_data()
        if self.my_local_rank == 0 and self._optimizer_ctx is not None:
            try:
                self.logger.info(f"{utcnow()} Stopping optimizer runtime context")
                self._optimizer_ctx.stop()
            except Exception as ex:
                self.logger.warning(f"{utcnow()} Failed to stop optimizer runtime context: {ex}")
            finally:
                self._optimizer_ctx = None
                self._optimizer_started = False
        self.comm.barrier()
        if dftracer_finalize and dftracer:
            self.args.finalize_dftracer(dftracer)

# Apply @tunable decorator to make_loader if dfoptimizer is available.
if _HAS_DFOPTIMIZER:
    # ---- ACTUATOR COSTS, seconds. Every value here is MEASURED on this machine; none is assumed.
    #
    # `cost_seconds` ranks candidates so the planner fires the cheapest that clears its gates. It is
    # a RANKING input, not the commit decision: the real cost of an expensive action is measured
    # while it runs, and staging abandons mid-copy if the observed rate contradicts the estimate
    # (seen in production -- 0.130 GB/s implied 2130 s against an 1089 s horizon, so it paid 265 s
    # and quit). Ranking needs order; committing needs the live number.
    #
    #   loader rebuild   0.206 s  mean of n=3 steady-state epochs, TF path, make_loader -> first
    #                             step (0.207/0.204/0.209; epoch 1 is 1.023 s and excluded as
    #                             first-touch). Applies to read_threads, prefetch_size and any other
    #                             knob whose reconcile is "rebuild the loader".
    #                             CAVEAT: TENSORFLOW ONLY. The PyTorch path rebuilds by FORKING
    #                             WORKERS, which is why the loader cache existed at all, and that
    #                             cost is UNMEASURED. Overriding via LIVEFLOW_REBUILD_COST is the
    #                             honest stopgap until it is measured.
    #   staging          465-1365 s over 6 runs of byte-identical work (465/474/477/679/1181/1365).
    #                             Declared at the LOW end: under-stating an expensive action is the
    #                             dangerous direction, so the planner faces the most favourable
    #                             number it could see and must still prefer a 0.2 s knob.
    #   restripe         ~4.4 h   258.5 GiB at a measured `lfs migrate` rate of 0.012 GB/s. Not
    #                             registered -- it cannot be actuated online at our run lengths.
    _REBUILD_COST = float(os.environ.get("LIVEFLOW_REBUILD_COST", "0.206"))
    _STAGE_COST = float(os.environ.get("LIVEFLOW_STAGE_COST", "465"))
    # 258.5 GiB at the measured `lfs migrate` rate of 0.012 GB/s. Registered rather than excluded:
    # whether an action is "offline only" is NOT a property of the action, it is a property of the
    # action's cost against the run's remaining length.
    #     3-epoch DLIO   ~25 min   ->  4.4 h is 1000%+ of the run   -> never
    #     50-epoch       ~7 h      ->  63%                          -> marginal
    #     LLM, 3 weeks   504 h     ->  0.9%                         -> trivially worth it
    # The same selection rule rejects it at 25 minutes and accepts it at 3 weeks, with no new logic:
    # gain * epochs_remaining has to clear the cost, and only the horizon changed.
    _RESTRIPE_COST = float(os.environ.get("LIVEFLOW_RESTRIPE_COST", "15840"))

    _knob_boundary = os.environ.get("DLIO_KNOB_BOUNDARY", "window_boundary")
    # data_source gets its OWN boundary, and it defaults to the epoch.
    #
    # THE BUG THIS FIXES, and it is silent. At window_boundary the first apply lands mid-epoch-1,
    # where `_last_epoch_seconds`, `_last_steps` and `_last_per_step_compute` are all still 0
    # because main.py sets them only after `_train` returns. staging_reconcile derives
    # `gain_per_epoch` from exactly those three, so gain=0, horizon=0, and the abort test
    # `horizon > cost_projected * 1.2` is false on the FIRST chunk. Every mid-epoch staging plan
    # aborts having paid for one chunk, and it reports that as a considered abort.
    #
    # It is also the wrong semantics. The measured decision (live4) was taken at the END of epoch 1
    # against a measured epoch: `ACTION epoch=1 stall=401.6s floor=115.0s remaining=3`. A
    # collective, expensive, semi-irreversible actuator belongs at an epoch boundary; the free
    # reversible ones stay at window boundaries and are unaffected by this.
    # Default flipped to the window boundary now that a mid-epoch horizon is measurable
    # (`_horizon_terms`). The original reason for pinning to the epoch was that mid-epoch the
    # horizon terms were all zero, so every mid-epoch stage aborted on its first chunk. That is no
    # longer true: elapsed time and completed steps give a projected epoch, and the horizon
    # contract gives the epochs after it. The pin cost the 32-node run its result -- the decision
    # was ready 60 s after the epoch 1 boundary and had to wait for the epoch 2 boundary, leaving
    # one epoch to repay an 884 s copy, which it correctly refused to do.
    # DLIO_DATA_SOURCE_BOUNDARY=epoch_boundary restores the old behaviour.
    _data_source_boundary = os.environ.get("DLIO_DATA_SOURCE_BOUNDARY", "window_boundary")
    _knob_min_threads = int(os.environ.get("DLIO_KNOB_MIN_THREADS", "0"))
    _knob_max_threads = int(os.environ.get(
        "DLIO_KNOB_MAX_THREADS", str(os.cpu_count() or 8)
    ))
    if _knob_min_threads > _knob_max_threads:
        _knob_min_threads = _knob_max_threads
    DLIOBenchmark.make_loader = tunable(knobs={
        "prefetch_size": knob(
            default=2, range=(1, 16), type=int, cost_seconds=_REBUILD_COST,
            responds_to={
                "dataloader_prefetch": {
                    "direction": "increase",
                    "step_mode": "add",
                    "step": 2,
                    "min_persistence": 1,
                    "cooldown_windows": 2,
                    "apply_when": _knob_boundary,
                },
            },
        ),
        # WHERE THE BYTES COME FROM, as a knob on the SAME path as read_threads/prefetch_size. It
        # rides the existing plan -> SHM mailbox -> reconcile machinery (verified on this branch: a
        # published plan moved read_threads 1 -> 2 -> 4 with acks returned). Nothing bespoke.
        #
        # The application does not know what any value MEANS or costs. "local" happens to be
        # satisfied by staging a shard to node-local NVMe (465-1365 s, collective, abortable), but
        # "replica", "compressed" and "cache" are the same kind of thing at different prices.
        # Resolvers are registered outside; cost and reversibility belong in the ActionDef the
        # OPTIMIZER reasons over, not here and not in the transport.
        "data_source": knob(
            # VALUES. "restriped" is opt-in (DFOPTIMIZER_REGISTER_RESTRIPED=1) because its resolver
            # ALWAYS declines -- it is registered and costed but not implemented, and its gain model
            # is unvalidated at one stream per file (gate v3: 1.40x at 16 streams, flat at 1,
            # p=0.74). As a declared level it is still an arm the acquisition explores, and being
            # never satisfied it stays under-sampled forever, so the gate never stops it. Measured
            # at 2 nodes: it won the epoch boundary twice, declined twice, and consumed the two
            # chances "local" had to fire in that epoch. Its correct deferral by cost is a replay
            # result and does not need a live arm to demonstrate.
            default="pfs", type=str,
            values=(["pfs", "local", "restriped"]
                    if os.environ.get("DFOPTIMIZER_REGISTER_RESTRIPED", "0") == "1"
                    else ["pfs", "local"]),
            # Cost per VALUE. Collapsing these to one number per knob would make "local" look as
            # expensive as a restripe, or a restripe look as cheap as a stage.
            cost_seconds={"pfs": 0.0, "local": _STAGE_COST, "restriped": _RESTRIPE_COST},
            # Staging relocates the corpus, and the objective on the other side of it is a
            # different quantity: a cosmoflow window went 37 s -> 5.6 s across the move. The
            # planner resets every other knob's samples when this one is applied.
            changes_regime=True,
            # And it relocates the bytes, so its gain is bounded by what the destination device can
            # deliver and its cost scales with how much there is to move (T3-0 section 4s).
            relocates_data=True,
            responds_to={
                "storage_starved": {
                    "direction": "set", "set_to": "local",
                    "min_persistence": 1, "cooldown_windows": 999,
                    "apply_when": _data_source_boundary,
                },
            },
        ),
        # Decode parallelism, OFF BY DEFAULT. `DFOPTIMIZER_REGISTER_COMPUTATION_THREADS=1` restores it.
        #
        # The comment here used to end "its effect model is therefore unvalidated, which is why
        # nothing selects it yet". Measured on Corona, 2 nodes, cosmoflow: something did select it,
        # in 44 of 46 windows, and it made the run 50% slower (epoch 1 at the shipped default
        # 79.95 s; epochs 2 and 3, after the knob had been driven to 1, 119.73 s and 122.94 s).
        #
        # The mechanism is domain ownership, not the knob. `_domain_of` has no entry for
        # `computation_threads`, so it falls to the default domain "compute" -- and on DLIO the
        # compute domain is a `time.sleep` (framework/torch_framework.py:43 -> utility.py:346;
        # tf_framework equivalently). `compute_time_proc_max` is therefore large and real (0.00551 s
        # per step, 11 ms of an 18.7 ms window here) and WINS the cross-layer argmax, but no
        # actuator anywhere can shorten a sleep. The planner then does the only thing it can with
        # the domain's sole knob: move it. On the TF path that knob is `num_parallel_calls` on the
        # decode map (tf_reader.py:103), so "tune compute" actually throttles decode, and driving it
        # to 1 serialises the map.
        #
        # Unregistering makes the compute domain own NO knob, which turns a compute-dominant window
        # into a DECLINE by exactly the mechanism the `unattributed` pseudo-domain already uses: a
        # domain that wins the argmax and owns nothing means there is nothing to do. That is the
        # correct answer for a simulated-compute benchmark, and it is a comparison rather than a
        # constant. It also stops a knob whose effect model is admittedly unvalidated from being
        # the most-actuated knob in the run.
        #
        # Two things to fix before turning it back on, recorded so this is a deferral and not a
        # deletion: `computation_threads` belongs in the DECODE domain, not compute, on any
        # framework where it drives a decode map; and DLIO's compute domain needs to be marked
        # actuator-less rather than merely knob-less, so the decline is explicit.
        **({"computation_threads": knob(
            default=1, range=(1, _knob_max_threads), type=int, cost_seconds=_REBUILD_COST,
            responds_to={
                "decode_pressure": {
                    "direction": "increase", "step_mode": "add", "step": 2,
                    "min_persistence": 1, "cooldown_windows": 2,
                    "apply_when": _knob_boundary,
                },
            },
        )} if os.environ.get("DFOPTIMIZER_REGISTER_COMPUTATION_THREADS", "0") == "1" else {}),
        # Read SHAPE, as opposed to read COUNT. AUTOTUNE cannot reach it (tf_reader.py:111 passes
        # transfer_size straight to TFRecordDataset while the AUTOTUNE block replaces only the three
        # parallelism args), but measured here it does not matter: +1.5%, p=0.32, n=3 at AUTOTUNE's
        # own operating point. Registered for completeness; nothing should select it on this
        # evidence.
        # NOT REGISTERED BY DEFAULT. `DFOPTIMIZER_REGISTER_TRANSFER_SIZE=1` restores it.
        #
        # It is not a parameter of `make_loader`, so `@tunable` cannot inject it and acks every plan
        # `knob_not_in_kwargs` (open bug #30, and BASELINE-HARNESS.md section 8 lists it as a static
        # knob only). Registering an unappliable knob is not harmless: a never-applied arm never
        # accumulates samples, so the acceptance gate treats it as under-sampled and lets it through
        # every window, and the cheapest-first escalation stops there. Measured on Corona at 2 nodes:
        # 144,048 rejected transfer_size plans and 0 applied ones in a single epoch, while
        # `data_source` -- ordered immediately behind it -- was never reached.
        #
        # The planner now also retires a knob after repeated refusals, so this would self-correct;
        # it is unregistered as well because the honest registry is the one that lists what the
        # application can actually do.
        **({"transfer_size": knob(
            default=262144, values=[262144, 1048576, 4194304, 16777216], type=int,
            cost_seconds=_REBUILD_COST,
            responds_to={},
        )} if os.environ.get("DFOPTIMIZER_REGISTER_TRANSFER_SIZE", "0") == "1" else {}),
        "read_threads": knob(
            default=max(0, _knob_min_threads), range=(_knob_min_threads, _knob_max_threads), type=int,
            cost_seconds=_REBUILD_COST,
            responds_to={
                "reader_parallelism": {
                    "direction": "increase",
                    "step_mode": "evidence",
                    "min_persistence": 1,
                    "cooldown_windows": 1,
                    "apply_when": _knob_boundary,
                },
                "reader_contention": {
                    "direction": "decrease",
                    "step": 1,
                    "min_persistence": 2,
                    "cooldown_windows": 3,
                    "apply_when": _knob_boundary,
                },
            },
        ),
    })(DLIOBenchmark.make_loader)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_benchmark(cfg: DictConfig):    
    benchmark = DLIOBenchmark(cfg['workload'])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()

def set_dftracer_initialize(status):
    global dftracer, dftracer_initialize, dftracer_finalize
    dftracer_initialize = status

def set_dftracer_finalize(status):
    global dftracer, dftracer_initialize, dftracer_finalize
    dftracer_finalize = status

def main() -> None:
    """
    The main method to start the benchmark runtime.
    """
    DLIOMPI.get_instance().initialize()
    run_benchmark()
    DLIOMPI.get_instance().finalize()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def query_config(cfg: DictConfig):
    DLIOMPI.get_instance().initialize()
    config = cfg['workload']
    
    value = None
    if "query" in config["workflow"]:
        key = config["workflow"]["query"]
        args = ConfigArguments.get_instance()
        LoadConfig(args, config)
        value = GetConfig(args, key)
    print(value) if value else print("None")
    DLIOMPI.get_instance().finalize()
    
if __name__ == '__main__':
    main()
    exit(0)
