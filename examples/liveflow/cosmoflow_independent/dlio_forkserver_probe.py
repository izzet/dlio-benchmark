"""Preload probe and lightweight warmup target; no DLIO/MPI/tracer imports."""
import json
import os
import pathlib
import sys
import time

def snapshot():
    maps=pathlib.Path('/proc/self/maps').read_text()
    return dict(pid=os.getpid(),parent_pid=os.getppid(),monotonic=time.monotonic(),
                heavy_modules={m:m in sys.modules for m in ['torch','h5py','numpy']},
                mpi_python='mpi4py.MPI' in sys.modules,
                mpi_mappings=sorted(set(line.split()[-1] for line in maps.splitlines()
                                        if any(s in line.lower() for s in ['libmpi','libmpich','mvapich']))),
                tracer_python=any(m=='dftracer' or m.startswith('dftracer.') for m in sys.modules),
                tracer_shm_mapped=bool(os.environ.get('LIVEFLOW_SHM_PREFIX') and os.environ['LIVEFLOW_SHM_PREFIX'] in maps),
                native_threads=len(list(pathlib.Path('/proc/self/task').iterdir())),
                hdf5_open_files=(sys.modules['h5py'].h5f.get_obj_count(types=sys.modules['h5py'].h5f.OBJ_FILE) if 'h5py' in sys.modules else 0))

state=snapshot()
if all(state['heavy_modules'].values()):
    assert not state['mpi_python'] and not state['mpi_mappings'],state
    assert not state['tracer_python'] and not state['tracer_shm_mapped'],state
    assert state['hdf5_open_files']==0 and state['native_threads']==1,state
    directory=pathlib.Path(os.environ['RUN_DIR'])/'forkserver-state'
    directory.mkdir(exist_ok=True)
    (directory/f'server-{os.getpid()}.json').write_text(json.dumps(state,indent=2)+'\n')

def warmup(connection):
    connection.send(snapshot())
    connection.close()
