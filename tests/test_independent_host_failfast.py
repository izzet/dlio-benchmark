#!/usr/bin/env python3
"""Offline launcher regression: fake Flux only; no jobs/resources submitted."""
import importlib.util,json,os,pathlib,subprocess,sys,tempfile,unittest
own=pathlib.Path(__file__).resolve().parents[1]/'examples/liveflow/cosmoflow_independent'
spec=importlib.util.spec_from_file_location('host_runner',own/'run-host-job.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
FAKE='''#!/usr/bin/env python3
import json,os,pathlib,sys,time
root=pathlib.Path(os.environ['FAKE_FLUX_ROOT']);args=sys.argv[1:]
if args[0]=='submit':
 host=next(x.split(':')[-1] for x in args if x.startswith('--requires='));job='f'+host
 (root/(job+'.command')).write_text(json.dumps(args));print(job)
elif args[:2]==['job','attach']:
 job=args[2]
 if job=='fbad':
  print(os.environ['FATAL_TEXT'],flush=True)
  if os.environ.get('IMMEDIATE')=='1':sys.exit(1)
  deadline=time.monotonic()+8
  while not (root/(job+'.canceled')).exists() and time.monotonic()<deadline:time.sleep(.05)
  sys.exit(1 if (root/(job+'.canceled')).exists() else 99)
 print('healthy host running',flush=True);time.sleep(2);print('healthy host complete',flush=True)
elif args[0]=='cancel':(root/(args[1]+'.canceled')).write_text('only this nested job')
else:sys.exit(98)
'''
class Tests(unittest.TestCase):
 def exercise(self,fatal,immediate=False):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp);binpath=root/'bin';binpath.mkdir();flux=binpath/'flux';flux.write_text(FAKE);flux.chmod(0o755)
   env=dict(os.environ,PATH=str(binpath)+':'+os.environ['PATH'],FAKE_FLUX_ROOT=tmp,FATAL_TEXT=fatal,IMMEDIATE=str(int(immediate)))
   procs=[];logs=[]
   for host in ['bad','good']:
    directory=root/host;directory.mkdir();log=(directory/'dlio.log').open('w');logs.append(log)
    procs.append(subprocess.Popen([sys.executable,str(own/'run-host-job.py'),str(directory),host,'app','--','true'],env=env,stdout=log,stderr=log))
   rc=[p.wait(timeout=12) for p in procs]
   for log in logs:log.close()
   self.assertEqual(rc,[1,0])
   self.assertFalse((root/'fgood.canceled').exists())
   if not immediate:self.assertTrue((root/'fbad.canceled').exists())
   failure=json.loads((root/'bad/app-failure.json').read_text());self.assertEqual(failure['nested_jobid'],'fbad');self.assertEqual(failure['reason'],fatal)
   self.assertIn('healthy host complete',(root/'good/dlio.log').read_text())
   self.assertFalse((root/'good/app-failure.json').exists())
   for job in ['fbad','fgood']:
    command=json.loads((root/(job+'.command')).read_text());self.assertIn('exit-on-error',command);self.assertIn('exit-timeout=5s',command)
 def test_mpi_signal11_other_host_continues(self):self.exercise('[corona263:mpi_rank_7][error_sighandler] Caught error: Segmentation fault (signal 11)')
 def test_worker_heap_corruption_other_host_continues(self):self.exercise('malloc(): unsorted double linked list corrupted')
 def test_immediate_failed_rank_records(self):self.exercise('DataLoader worker (pid 1894750) is killed by signal: Segmentation fault.',True)
 def test_retained_failure_strings(self):
  for text in ['[corona263:mpi_rank_7][error_sighandler] Caught error: Segmentation fault (signal 11)', 'malloc(): unsorted double linked list corrupted']:
   self.assertIsNotNone(runner.fatal_line(text))
if __name__=='__main__':unittest.main()
