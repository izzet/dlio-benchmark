"""A native crash excludes a whole host pair, preserves raw data, and caps loss."""
import importlib.util,json,pathlib,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=ROOT/'examples/liveflow/cosmoflow_independent/paired-hosts.py';spec=importlib.util.spec_from_file_location('paired_hosts',p);policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
class Tests(unittest.TestCase):
 def prepare(self,tmp):
  root=pathlib.Path(tmp);(root/'hosts.txt').write_text(''.join(f'h{i}\n' for i in range(16)));return root
 def failure(self,root,arm,host,native=True):
  d=root/arm/host;d.mkdir(parents=True,exist_ok=True)
  reason='Segmentation fault (signal 11)' if native else 'configuration rejected'
  (d/'dlio.log').write_text(reason);(d/'app-failure.json').write_text(json.dumps(dict(reason=reason,host=host)))
 def test_two_crashes_keep_fourteen_pairs_and_raw_completed_arm(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=self.prepare(tmp);raw=root/'original/h0/keep.json';raw.parent.mkdir(parents=True);raw.write_text('retained prior successful arm')
   self.failure(root,'static-best','h0');policy.exclude(root,'static-best','h0')
   self.failure(root,'liveflow','h1');eligible=policy.exclude(root,'liveflow','h1')
   self.assertEqual(set(eligible),{f'h{i}' for i in range(2,16)});self.assertTrue(raw.exists())
   self.assertEqual({r['host'] for r in json.loads((root/'excluded-hosts.json').read_text())},{'h0','h1'})
 def test_third_crash_invalidates_block_and_remains_disclosed(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=self.prepare(tmp)
   for h in ['h0','h1']:self.failure(root,'original',h);policy.exclude(root,'original',h)
   self.failure(root,'original','h2')
   with self.assertRaisesRegex(ValueError,'only 13 paired'):policy.exclude(root,'original','h2')
   self.assertEqual(len(json.loads((root/'excluded-hosts.json').read_text())),3)
 def test_non_native_failure_never_silently_excludes(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=self.prepare(tmp);self.failure(root,'original','h0',False)
   with self.assertRaisesRegex(ValueError,'non-native'):policy.exclude(root,'original','h0')
   self.assertFalse((root/'excluded-hosts.json').exists())
 def test_duplicate_reports_do_not_drop_another_pair(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=self.prepare(tmp);self.failure(root,'original','h0');policy.exclude(root,'original','h0');self.assertEqual(len(policy.exclude(root,'original','h0')),15)
 def test_exact_terminal_hdf5_filter_error_excludes_same_pair(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=self.prepare(tmp);self.failure(root,'maxresource-adapted','h0',False)
   d=root/'maxresource-adapted/h0'
   (d/'dlio.log').write_text('OSError: Caught OSError in DataLoader worker process 2.\nOriginal Traceback (most recent call last):\n'+policy.runner.HDF5_FILTER_FAILURE+'\nflux-job: task(s) exited with exit code 1\n')
   self.assertEqual(len(policy.exclude(root,'maxresource-adapted','h0')),15)
   record=json.loads((root/'excluded-hosts.json').read_text())[0]
   self.assertEqual(record['failure_class'],'hdf5_filter_read_error')
   self.assertEqual(record['host'],'h0')
 def test_other_worker_oserrors_and_filter_mentions_stop(self):
  cases=['OSError: Caught OSError in DataLoader worker process 2.\nOSError: disk read failed',
         'OSError: Caught OSError in DataLoader worker process 2.\n'+policy.runner.HDF5_FILTER_FAILURE+' extra text',
         'warning: '+policy.runner.HDF5_FILTER_FAILURE,
         policy.runner.HDF5_FILTER_FAILURE]
  for text in cases:
   with self.subTest(text=text),tempfile.TemporaryDirectory() as tmp:
    root=self.prepare(tmp);self.failure(root,'original','h0',False)
    (root/'original/h0/dlio.log').write_text(text)
    with self.assertRaisesRegex(ValueError,'non-native'):policy.exclude(root,'original','h0')
    self.assertFalse((root/'excluded-hosts.json').exists())
if __name__=='__main__':unittest.main()
