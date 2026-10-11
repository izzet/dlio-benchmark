"""DLIO-PyTorch MPI thread signal isolation for forked reader retirement."""
import signal
import sys


def initialize_mpi(initialize):
    """Keep SIGCHLD out of MPI-created polling threads, not out of PyTorch.

    New native threads inherit the creator's mask. Block only while MPI starts
    its threads, then restore the caller's exact mask. DataLoader's main-thread
    SIGCHLD handler and waitpid/join remain active for every pool and epoch.
    Masking only around shutdown is too late to protect an existing MPI thread.
    """
    mpi = sys.modules.get("mpi4py.MPI")
    if mpi is not None and mpi.Is_initialized():
        raise RuntimeError("DLIO worker signal isolation must precede MPI initialization")
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD})
    try:
        return initialize()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
