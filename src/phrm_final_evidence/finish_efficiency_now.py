import os,signal,time,json
from pathlib import Path
from common import ROOT,write_json,log
from dispatch import call,activity
pid=int((ROOT/'logs/master.pid').read_text())
assert Path(f'/proc/{pid}/cwd').resolve()==ROOT
assert b'dispatch.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
retry=3905373
try:
    if Path(f'/proc/{retry}/cwd').resolve()==ROOT and b'retry_efficiency.py' in Path(f'/proc/{retry}/cmdline').read_bytes():os.kill(retry,signal.SIGTERM)
except FileNotFoundError:pass
os.kill(pid,signal.SIGSTOP)
log('OWN_QUEUE_SCHEDULING_PAUSED',pid=pid,current_gpu_children_continue=True)
try:
    limit=time.monotonic()+300
    while activity(3):
        if time.monotonic()>limit:raise RuntimeError('GPU3 did not become idle; profile postponed')
        time.sleep(3)
    for name in ('stid','stgformer'):
        for mode in ('base','phrm'):
            rc=call(3,f'efficiency_LargeSTSD_{name}_{mode}_completedinputs',['largest_efficiency','--backbone',name,'--mode',mode],True)
            if rc:raise RuntimeError('completed-input efficiency profile failed')
    write_json(ROOT/'audit/efficiency_completion.json',dict(complete=True,profiles=8,input_fix='pinned vendor YAML completed',current_training_code_changed=False))
finally:
    os.kill(pid,signal.SIGCONT);log('OWN_QUEUE_SCHEDULING_RESUMED',pid=pid)
