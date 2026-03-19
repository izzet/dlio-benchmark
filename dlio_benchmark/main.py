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
from time import time
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
        self.comm = DLIOMPI.get_instance().comm()
        self.my_rank = self.args.my_rank = DLIOMPI.get_instance().rank()
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
        self._ensure_optimizer_runtime_started()
        self.comm.barrier()

    def _ensure_optimizer_runtime_started(self):
        if (
            self._optimizer_started
            or not _HAS_DFOPTIMIZER
            or os.environ.get("DFOPTIMIZER_ENABLE", "0") != "1"
            or self.my_rank != 0
        ):
            return

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
        except Exception as ex:
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

    def make_loader(self, epoch, prefetch_size=None, read_threads=None):
        """Create or reuse a training DataLoader for the given epoch.

        When prefetch_size and read_threads are unchanged from the previous
        epoch, the cached DataLoader (with its persistent workers) is reused
        to avoid per-epoch fork/init overhead.  A new DataLoader is created
        only when knob values actually change.

        Rank 0 receives overrides from the @tunable decorator; values are
        then broadcast to all ranks so every process uses the same settings.
        """
        if prefetch_size is None:
            prefetch_size = self.args.prefetch_size
        if read_threads is None:
            read_threads = self.args.read_threads

        # Broadcast knob values from rank 0 to ensure consistency
        knobs = [prefetch_size, read_threads]
        knobs = self.comm.bcast(knobs, root=0)
        prefetch_size, read_threads = knobs

        self.args.prefetch_size = prefetch_size
        self.args.read_threads = read_threads

        # Reuse cached loader if knobs haven't changed
        if (
            self._cached_loader is not None
            and self._cached_prefetch == prefetch_size
            and self._cached_read_threads == read_threads
        ):
            if self.my_rank == 0:
                self.logger.output(
                    f"{utcnow()} make_loader: epoch={epoch} prefetch={prefetch_size}"
                    f" threads={read_threads} (cached)"
                )
            loader = self._cached_loader
        else:
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

        self._ensure_optimizer_runtime_started()

        return loader

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
        while True:
            dft_ai.update(epoch=epoch, step=overall_step, args={"max_steps": max_steps})
            # Window boundary (gated by every_n) — replaces pipeline.step
            # control events. Only emits to Mofka at cadence boundaries.
            if self._window is not None:
                self._window.start()
            dft_ai.pipeline.step.start(metadata=True)
            try:
                batch = next(batch_iter)
            except StopIteration:
                dft_ai.pipeline.step.stop(metadata=True)
                if self._window is not None:
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
                    self._window.stop()
                break
            self.stats.batch_loaded(epoch, overall_step, block)
            dft_ai.compute.step.start(metadata=True)
            try:
                computation_time = self.args.computation_time
                if (isinstance(computation_time, dict) and len(computation_time) > 0) or (isinstance(computation_time, float) and  computation_time > 0):
                    self.framework.trace_object("Train", overall_step, 1)
                self.stats.start_compute()
                self.framework.compute(batch, epoch, block_step, self.computation_time)
                self.stats.batch_processed(epoch, overall_step, block)
                # This is the barrier to simulate allreduce. It is required to simulate the actual workloads.
                self.comm.barrier()
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
                dft_ai.compute.step.stop(metadata=True)
            dft_ai.pipeline.step.stop(metadata=True)
            if self._window is not None:
                self._window.stop()
            overall_step += 1
            # start a new block here
            if block_step == 1 and block != 1:
                self.stats.start_block(epoch, block)
            self.stats.start_loading()

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
            # Start at min(1000, steps_per_epoch) to keep drain manageable
            # (~90K events per window at 90 events/step). The optimizer
            # can adjust cadence via the ack mechanism.
            _steps_per_epoch = math.floor(
                self.num_samples * self.num_files_train / self.batch_size / self.comm_size
            )
            _initial_every_n = min(1000, _steps_per_epoch)
            if _HAS_DFOPTIMIZER and dftracer:
                self._window = Window(
                    dftracer.get_instance(),
                    max_every_n=_steps_per_epoch,
                )
                self._window.update_cadence(_initial_every_n)
            else:
                self._window = None

            self.comm.barrier()
            for epoch in dft_ai.pipeline.epoch.iter(range(1, self.epochs + 1), include_iter=False):
                dft_ai.pipeline.epoch.start(metadata=True)
                self.stats.start_epoch(epoch)
                self.next_checkpoint_step = self.steps_between_checkpoints
                # Create fresh train loader each epoch (subject to @tunable overrides)
                loader = self.make_loader(
                    epoch=epoch,
                    prefetch_size=self.args.prefetch_size,
                    read_threads=self.args.read_threads,
                )
                self.stats.start_train(epoch)
                steps = self._train(epoch, loader)
                self.stats.end_train(epoch, steps)
                # Force window boundary at epoch end so analyzer processes
                # any accumulated events before the next epoch begins.
                if self._window is not None:
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
        if self.my_rank == 0 and self._optimizer_ctx is not None:
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
    DLIOBenchmark.make_loader = tunable(knobs={
        "prefetch_size": knob(
            default=2, range=(1, 16), type=int,
            responds_to={
                "dataloader_prefetch": {
                    "direction": "increase",
                    "step": 2,
                    "min_persistence": 2,
                    "cooldown_windows": 4,
                },
            },
        ),
        "read_threads": knob(
            default=0, range=(0, 16), type=int,
            responds_to={
                "reader_parallelism": {
                    "direction": "increase",
                    "step": 1,
                    "min_persistence": 2,
                    "cooldown_windows": 4,
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
