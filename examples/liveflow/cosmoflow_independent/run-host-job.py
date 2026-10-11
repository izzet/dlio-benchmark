#!/usr/bin/env python3
"""Launch and monitor only one host's nested Flux job, never its allocation."""
import argparse
import json
import os
import pathlib
import re
import subprocess
import time

FATAL = re.compile(r'unsorted double linked list corrupted|Unexpected segmentation fault|Segmentation fault|signal 11|Bus error|DataLoader worker.*(?:killed|exited unexpectedly)', re.I)
HDF5_FILTER_FAILURE = "OSError: Can't synchronously read data (filter returned failure during read)"

def fatal_line(text):
    return next((line for line in text.splitlines() if FATAL.search(line)), None)

def pair_exclusion_reason(text):
    """User-approved allowlist: native crash or exact terminal HDF5 filter error."""
    native = fatal_line(text)
    if native:
        return native
    lines = text.splitlines()
    worker_exception = any(re.fullmatch(r'OSError: Caught OSError in DataLoader worker process \d+\.', line.strip()) for line in lines)
    if worker_exception:
        return next((line for line in lines if line.strip() == HDF5_FILTER_FAILURE), None)
    return None

def record_failure(directory, host, kind, jobid, reason, returncode=None):
    record = dict(unix_s=time.time(), host=host, kind=kind, nested_jobid=jobid,
                  reason=reason, returncode=returncode, cancellation_scope='this nested host job only')
    (directory / f'{kind}-failure.json').write_text(json.dumps(record, indent=2)+'\n')
    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=pathlib.Path)
    parser.add_argument('host')
    parser.add_argument('kind', choices=['app', 'agent'])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ['--']: command = command[1:]
    if not command: parser.error('missing command')
    directory = args.directory
    options = ['-o', 'exit-on-error', '-o', 'exit-timeout=5s']
    cmd = ['flux', 'submit', '--flags=waitable', f'--requires=host:{args.host}',
           '-N1', '-n8' if args.kind == 'app' else '-n1',
           '-c5' if args.kind == 'app' else '-c1',
           '-o', 'cpu-affinity=off', '-o', 'mpibind=off', *options, *command]
    if args.kind == 'app':
        cmd[2:2] = [f'--cwd={directory.resolve()}']
        if os.environ.get('DLIO_CORE_LIMIT_BYTES'):
            cmd[2:2] = ['--rlimit=core='+os.environ['DLIO_CORE_LIMIT_BYTES']]
    (directory / f'{args.kind}-submission-command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    jobid = subprocess.check_output(cmd, text=True).strip()
    (directory / f'{args.kind}-jobid.txt').write_text(jobid+'\n')
    attach = subprocess.Popen(['flux', 'job', 'attach', jobid])
    logfile = directory / ('dlio.log' if args.kind == 'app' else 'agents.log')
    offset = 0
    pending = ''
    detected = False
    deadline=time.monotonic()+float(os.environ.get('DLIO_APP_TIMEOUT_S','0'))
    while attach.poll() is None:
        if args.kind == 'app' and os.environ.get('DLIO_APP_TIMEOUT_S') and time.monotonic()>deadline:
            detected=True
            record_failure(directory,args.host,args.kind,jobid,'application diagnostic timeout')
            subprocess.run(['flux','cancel',jobid],check=False)
            break
        if args.kind == 'app' and logfile.exists():
            with logfile.open(errors='replace') as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            lines = (pending+chunk).splitlines(keepends=True)
            pending = lines.pop() if lines and not lines[-1].endswith('\n') else ''
            line = fatal_line(''.join(lines)+pending)
            if line:
                detected = True
                record_failure(directory, args.host, args.kind, jobid, line)
                subprocess.run(['flux', 'cancel', jobid], check=False)
                break
        time.sleep(1)
    rc = attach.wait()
    if detected and rc == 0: rc = 1
    if rc and not (directory / f'{args.kind}-failure.json').exists():
        reason = pair_exclusion_reason(logfile.read_text(errors='replace')) if logfile.exists() else None
        record_failure(directory, args.host, args.kind, jobid, reason or 'nested job exited unsuccessfully', rc)
    return rc

if __name__ == '__main__':
    raise SystemExit(main())
