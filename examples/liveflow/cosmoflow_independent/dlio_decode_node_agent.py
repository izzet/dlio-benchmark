#!/usr/bin/env python3
"""DLIO-only project entry point; install after the real agent's bootstrap."""
import sys
import pathlib
import node_agent

assert pathlib.Path(node_agent.__file__).resolve() == pathlib.Path("/p/vast1/izzet/lfmr/liveflow-campaign-v4.0.6-agentperf-20261010/source/liveflow_shm/python/node_agent.py")


def main():
    original_bootstrap = node_agent.bootstrap

    def bootstrap_with_adapter(*args, **kwargs):
        original_bootstrap(*args, **kwargs)
        # Bootstrap evicts analyzer modules. Import/install only after it finishes.
        from dlio_pytorch_decode_adapter import install
        install()

    node_agent.bootstrap = bootstrap_with_adapter
    try:
        return node_agent.main()
    finally:
        node_agent.bootstrap = original_bootstrap


if __name__ == '__main__':
    sys.exit(main())
