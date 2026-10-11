#!/usr/bin/env python3
"""Disclose native host crashes and remove the same host from every arm pair."""
import argparse,importlib.util,json,pathlib,time
own=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('host_runner',own/'run-host-job.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
def exclude(root,arm,host,minimum=14):
 hosts=(root/'hosts.txt').read_text().splitlines();assert len(hosts)==16 and len(set(hosts))==16
 assert host in hosts and 14<=minimum<=16
 directory=root/arm/host;failure=directory/'app-failure.json'
 reason=runner.fatal_line((directory/'dlio.log').read_text(errors='replace')) if (directory/'dlio.log').exists() else None
 if not reason or not failure.exists():raise ValueError('non-native or unrecorded failure; block must stop')
 output=root/'excluded-hosts.json';records=json.loads(output.read_text()) if output.exists() else []
 if host not in {r['host'] for r in records}:
  records.append(dict(host=host,failed_arm=arm,reason=reason,failure=json.loads(failure.read_text()),unix_s=time.time(),policy='exclude same host from all arms in this block; retain all raw data'))
 output.write_text(json.dumps(records,indent=2)+'\n')
 eligible=[h for h in hosts if h not in {r['host'] for r in records}]
 (root/'paired-hosts.txt').write_text(''.join(h+'\n' for h in eligible))
 if len(eligible)<minimum:raise ValueError(f'only {len(eligible)} paired hosts; minimum {minimum}; block invalid')
 return eligible
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=pathlib.Path);p.add_argument('arm');p.add_argument('host');p.add_argument('--minimum',type=int,default=14);a=p.parse_args()
 try:print(json.dumps(dict(paired_hosts=exclude(a.root,a.arm,a.host,a.minimum))))
 except ValueError as e:p.exit(1,str(e)+'\n')
