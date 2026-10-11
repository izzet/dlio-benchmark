"""Picklable DLIO dataset; importing this module never initializes MPI."""
import json
import os
import pathlib
import sys
import time

# DLIO worker configuration uses parent rank metadata, not an MPI communicator.
# Set this before DLIO imports; leave an already-initialized parent untouched.
if 'mpi4py.MPI' not in sys.modules:
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False

from dlio_benchmark.data_loader.torch_data_loader import TorchDataset

class IndexedTorchDataset(TorchDataset):
    def worker_init(self, worker_id):
        started = time.monotonic()
        mpi_before = 'mpi4py.MPI' in sys.modules
        hdf_before = 'h5py' in sys.modules
        super().worker_init(worker_id)
        from dlio_benchmark.utils.config import ConfigArguments
        args = ConfigArguments.get_instance()
        directory = pathlib.Path(os.environ['RUN_DIR'])/'worker-lifecycle'
        directory.mkdir(exist_ok=True)
        record = dict(pid=os.getpid(), parent_pid=os.getppid(), rank=args.my_rank,
                      epoch=self.epoch_number, worker_id=worker_id,
                      workers=self.num_workers, context=args.multiprocessing_context,
                      init_start_monotonic=started, ready_monotonic=time.monotonic(),
                      unix_s=time.time(), mpi_module_before_init=mpi_before,
                      mpi_module_after_init='mpi4py.MPI' in sys.modules,
                      h5py_before_init=hdf_before, h5py_after_init='h5py' in sys.modules)
        if os.environ.get('DLIO_LF_ISOLATE_CORES') == '2':
            record['cpus'] = sorted(os.sched_getaffinity(0))
            record['agent_cpulist'] = os.environ['LIVEFLOW_AGENT_CPUSET']
        (directory/f'worker-{os.getpid()}.json').write_text(json.dumps(record)+'\n')

    def __getitem__(self, index):
        return super().__getitem__(index), index
