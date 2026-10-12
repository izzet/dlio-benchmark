"""Qualify the actual launcher's default and explicit topology-discovery branch."""
import os,pathlib,subprocess,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'examples/liveflow/cosmoflow_independent/run-arm-independent-campaign.sh'
class Tests(unittest.TestCase):
 def branch(self,arm,value=None):
  source=SCRIPT.read_text();source=source[source.index('# Isolation is an opt-in'):source.index("agent_pid=''")]
  with tempfile.TemporaryDirectory() as tmp:
   env=dict(os.environ,ARM=arm,agentpy='python',OWN='/cell',RUN_DIR=tmp,LIVEFLOW_AGENT_CPUSET='inherited-stale')
   env.pop('DLIO_LF_ISOLATE_CORES',None)
   if value is not None:env['DLIO_LF_ISOLATE_CORES']=value
   setup='set -euo pipefail\nhostrun(){ echo called >> "$RUN_DIR/calls"; echo 23,47,71,95; }\n'
   result=subprocess.run(['bash','-c',setup+source+'printf "isolation=%s\\ncpuset=%s\\n" "${DLIO_LF_ISOLATE_CORES-unset}" "${LIVEFLOW_AGENT_CPUSET-unset}"'],env=env,text=True,capture_output=True)
   return result,(pathlib.Path(tmp)/'calls').exists()
 def test_liveflow_default_clears_stale_mask_without_discovery(self):
  r,called=self.branch('liveflow');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout,'isolation=unset\ncpuset=unset\n');self.assertFalse(called)
 def test_liveflow_explicit_zero_stays_off(self):
  r,called=self.branch('liveflow','0');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout,'isolation=unset\ncpuset=unset\n');self.assertFalse(called)
 def test_liveflow_two_is_opt_in(self):
  r,called=self.branch('liveflow','2');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout,'isolation=2\ncpuset=23,47,71,95\n');self.assertTrue(called)
 def test_baseline_always_clears_isolation(self):
  for arm in ['original','maxresource-adapted','static-best']:
   with self.subTest(arm=arm):
    r,called=self.branch(arm,'2');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout,'isolation=unset\ncpuset=unset\n');self.assertFalse(called)
 def test_invalid_option_stops(self):
  r,called=self.branch('liveflow','1');self.assertEqual(r.returncode,2);self.assertFalse(called)
if __name__=='__main__':unittest.main()
