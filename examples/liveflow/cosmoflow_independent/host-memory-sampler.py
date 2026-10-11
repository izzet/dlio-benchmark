#!/usr/bin/env python3
"""Opt-in /proc observer pinned to reserved agent cores; no app imports or IPC."""
import json
import os
import pathlib
import socket
import sys
import time

root = pathlib.Path(sys.argv[1])
mask = set()
for part in os.environ['LIVEFLOW_AGENT_CPUSET'].split(','):
    ends = part.split('-')
    mask.update(range(int(ends[0]), int(ends[-1]) + 1))
os.sched_setaffinity(0, mask)
assert os.sched_getaffinity(0) == mask
print(json.dumps(dict(event='sampler_start', host=socket.gethostname(), pid=os.getpid(),
                     cpus=sorted(mask), interval_s=10, rss_note='RSS sum double-counts shared pages; not PSS')), flush=True)
while not (root / 'host-memory.stop').exists():
    started = time.monotonic()
    cpu_started = time.process_time()
    mem = {}
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in {'MemTotal', 'MemAvailable', 'MemFree', 'Shmem', 'SwapFree'}:
            mem[key + '_bytes'] = int(value.split()[0]) * 1024
    processes = {}
    for entry in pathlib.Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            processes[int(entry.name)] = dict(ppid=int(stat[1]), rss_bytes=int(stat[21])*os.sysconf('SC_PAGE_SIZE'))
        except (OSError, ValueError, IndexError):
            pass
    roots = set()
    for binding in (root / 'rank-bindings').glob('rank-*.json'):
        try:
            roots.add(json.loads(binding.read_text())['pid'])
        except (OSError, ValueError, KeyError):
            pass
    app_pids = set(roots)
    while True:
        extra = {pid for pid, row in processes.items() if row['ppid'] in app_pids} - app_pids
        if not extra:
            break
        app_pids.update(extra)
    app = {str(pid): processes[pid]['rss_bytes'] for pid in sorted(app_pids) if pid in processes}
    shm = os.statvfs('/dev/shm')
    print(json.dumps(dict(event='sample', unix_s=time.time(), monotonic=time.monotonic(),
                         host=socket.gethostname(), **mem,
                         shm_capacity_bytes=shm.f_blocks*shm.f_frsize,
                         shm_used_bytes=(shm.f_blocks-shm.f_bfree)*shm.f_frsize,
                         shm_available_bytes=shm.f_bavail*shm.f_frsize,
                         app_rss_sum_bytes=sum(app.values()), app_pid_rss_bytes=app,
                         sampler_cpu_s=time.process_time()-cpu_started,
                         sampler_wall_s=time.monotonic()-started)), flush=True)
    deadline=started+10
    while time.monotonic()<deadline and not (root / 'host-memory.stop').exists():
        time.sleep(min(0.5, max(0, deadline-time.monotonic())))
print(json.dumps(dict(event='sampler_stop', unix_s=time.time())), flush=True)
