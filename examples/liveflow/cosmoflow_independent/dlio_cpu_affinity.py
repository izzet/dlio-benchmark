"""LF-only topology-derived CPU partition; no application/native imports."""
import json
import os
from pathlib import Path
import socket


def cpulist(cpus):
    return ','.join(str(c) for c in sorted(cpus))


def topology():
    allowed = sorted(os.sched_getaffinity(0))
    groups = {}
    rows = []
    for cpu in allowed:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        top = base / 'topology'
        socket_id = int((top / 'physical_package_id').read_text())
        core_id = int((top / 'core_id').read_text())
        nodes = list(base.glob('node[0-9]*'))
        node = int(nodes[0].name[4:]) if nodes else None
        siblings = (top / 'thread_siblings_list').read_text().strip()
        groups.setdefault((socket_id, core_id), []).append(cpu)
        rows.append(dict(cpu=cpu, socket=socket_id, core=core_id, numa=node,
                         hardware_siblings=siblings))
    physical = sorted(groups, key=lambda key: min(groups[key]))
    return dict(host=socket.gethostname(), online=Path('/sys/devices/system/cpu/online').read_text().strip(),
                allowed=allowed, physical_cores=len(physical), rows=rows), groups, physical


def partition():
    info, groups, physical = topology()
    # The shipped wrapper partitions the first48 physical cores into8 blocks of6.
    assert len(physical) == 48, ('Expected Corona48 physical cores', info)
    sockets = sorted({key[0] for key in physical})
    assert len(sockets) == 2, ('Expected Corona two sockets', info)
    reserved = {max((key for key in physical if key[0] == s), key=lambda k: min(groups[k]))
                for s in sockets}
    agent = sorted(c for key in reserved for c in groups[key])
    # Preserve the shipped wrapper's ONE logical CPU per physical core.
    ranks = [sorted(min(groups[key]) for key in physical[r*6:(r+1)*6] if key not in reserved)
             for r in range(8)]
    app = sorted(c for rank in ranks for c in rank)
    assert not set(agent) & set(app)
    assert len(app) == 46
    assert len({key for key in physical if key not in reserved}) == 46
    assert all(ranks) and len(set(app)) == len(app)
    return dict(topology=info, agent_cpus=agent, agent_cpulist=cpulist(agent),
                agent_physical_cores=2, app_cpus=app, app_physical_cores=46,
                rank_cpus=ranks, unused_allowed_cpus=sorted(set(info['allowed'])-set(agent)-set(app)),
                app_smt_policy='shipped one logical CPU per physical core; agent online siblings allowed',
                rank_physical_cores=[sum(key not in reserved for key in physical[r*6:(r+1)*6]) for r in range(8)])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    plan = partition()
    a.output.write_text(json.dumps(plan, indent=2)+'\n')
    print(plan['agent_cpulist'])


if __name__ == '__main__':
    main()
