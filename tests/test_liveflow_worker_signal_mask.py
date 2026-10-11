"""Signal-mask restoration matters to PyTorch's worker retirement handler."""
import importlib.util,pathlib,signal,sys,types,unittest
from unittest.mock import patch
path=pathlib.Path(__file__).resolve().parents[1]/'dlio_benchmark/utils/liveflow_worker_lifecycle.py'
spec=importlib.util.spec_from_file_location('worker_signal_mask',path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
class Tests(unittest.TestCase):
 def exercise(self,raises):
  before=signal.pthread_sigmask(signal.SIG_BLOCK,set())
  observed=[]
  def initialize():
   observed.append(signal.pthread_sigmask(signal.SIG_BLOCK,set()))
   if raises:raise ValueError('initializer failed')
   return 'initialized'
  with patch.dict(sys.modules,{'mpi4py.MPI':types.SimpleNamespace(Is_initialized=lambda:False)}):
   if raises:
    with self.assertRaisesRegex(ValueError,'initializer failed'):helper.initialize_mpi(initialize)
   else:self.assertEqual(helper.initialize_mpi(initialize),'initialized')
  self.assertIn(signal.SIGCHLD,observed[0]);self.assertEqual(before,signal.pthread_sigmask(signal.SIG_BLOCK,set()))
 def test_success_restores_exact_mask(self):self.exercise(False)
 def test_exception_restores_exact_mask(self):self.exercise(True)
 def test_rejects_late_isolation(self):
  with patch.dict(sys.modules,{'mpi4py.MPI':types.SimpleNamespace(Is_initialized=lambda:True)}):
   with self.assertRaisesRegex(RuntimeError,'precede MPI'):helper.initialize_mpi(lambda:self.fail('must not run'))
if __name__=='__main__':unittest.main()
