"""CPU-only check of the cell's indexed payload and RNG under optional timers."""
import json,os,pathlib,subprocess,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
 def test_fork_reader_payload_rng_and_order(self):
  with tempfile.TemporaryDirectory() as tmp:
   results=[]
   for enabled in ['0','1']:
    env=dict(os.environ,PYTHONPATH=str(ROOT),LIVEFLOW_MOTIVATION_TIMERS=enabled,LIVEFLOW_MOTIVATION_CAPTURE_STEPS='6',LIVEFLOW_MOTIVATION_WARMUP_STEPS='2',LIVEFLOW_MOTIVATION_OUTPUT_DIR=tmp,LIVEFLOW_MOTIVATION_GPU_TRACE='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    out=subprocess.check_output([sys.executable,__file__,'--child'],env=env,text=True)
    results.append(json.loads(out.strip().splitlines()[-1]))
   self.assertEqual(results[0],results[1]);files=list(pathlib.Path(tmp).glob('*.jsonl'));self.assertEqual(len(files),1)
   rows=[json.loads(line) for line in files[0].read_text().splitlines()];self.assertEqual(rows[0]['recorded_steps'],6)
   self.assertEqual([r['consumed_sample_ids'] for r in rows[1:]],[ [i] for i in range(2,8)])
   self.assertTrue(all(r['next_begin_ns']>r['begin_ns'] for r in rows[1:]))
if __name__=='__main__':
 if '--child' in sys.argv:
  import hashlib,numpy as np,torch
  from dlio_benchmark.data_loader import motivation_timers as mot
  class IndexedDataset:
   def __len__(self):return 16
   def __getitem__(self,index):return np.full((8,8),index,dtype=np.uint8),index
  torch.manual_seed(19);np.random.seed(19)
  dataset,collate=mot.loader_inputs(IndexedDataset())
  loader=torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=1,multiprocessing_context='fork',collate_fn=collate)
  payload=[];indices=[]
  iterator=iter(loader);mot.wrap_get_data(iterator)
  for batch in mot.observe(iterator,1):
   payload.append(batch[0].numpy().tobytes());indices.extend(batch[1].tolist())
  mot.flush()
  print(json.dumps(dict(payload_sha256=hashlib.sha256(b''.join(payload)).hexdigest(),indices=indices,torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),numpy_rng_sha256=hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest())))
 else:unittest.main()
