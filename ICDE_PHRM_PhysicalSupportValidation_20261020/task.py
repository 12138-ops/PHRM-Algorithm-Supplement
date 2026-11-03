"""One lock-protected preflight and resumable fit, retaining all failed logs."""
import sys,subprocess,fcntl
from common import ROOT

job=sys.argv[1];folder=ROOT/'results'/job;folder.mkdir(parents=True,exist_ok=True)
with (folder/'task.lock').open('a') as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit(75)
 for extra in [['--preflight'],[]]:
  command=[sys.executable,'-B','-u',str(ROOT/'fit_support.py'),'--job',job]+extra
  code=subprocess.call(command,cwd=ROOT)
  if code:raise SystemExit(code)
