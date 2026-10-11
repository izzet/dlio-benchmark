"""Flux executes a copied batch script; entry paths must use the pinned checkout."""
import os,pathlib,shutil,subprocess,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
 def test_copied_forkserver_entry_uses_checkout(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp);fake=root/'python3';fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');fake.chmod(0o755)
   script=root/'script';shutil.copyfile(ROOT/'examples/liveflow/cosmoflow_independent/forkserver-test.sh',script)
   env=dict(os.environ,PATH=tmp+':'+os.environ['PATH'],DLIO_CAMPAIGN_CHECKOUT=str(ROOT))
   out=subprocess.check_output(['bash',str(script)],env=env,text=True)
   self.assertIn(str(ROOT/'examples/liveflow/cosmoflow_independent/forkserver-test.py'),out)
 def test_copied_block_prefix_uses_checkout(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=pathlib.Path(tmp)/'script'
   source=(ROOT/'examples/liveflow/cosmoflow_independent/independent-campaign-block.sh').read_text()
   p.write_text(source[:source.index('block=${1:?}')]+"printf '%s\\n' \"$OWN\"\n")
   env=dict(os.environ,DLIO_CAMPAIGN_CHECKOUT=str(ROOT))
   self.assertEqual(subprocess.check_output(['bash',str(p)],env=env,text=True).strip(),str(ROOT/'examples/liveflow/cosmoflow_independent'))
if __name__=='__main__':unittest.main()
