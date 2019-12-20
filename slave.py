import time
from mujoco_parallel import WorkerManager

# Start workers
wm = WorkerManager()
wm.start_redis()
wm.create_workers()

# Wait for eternity
while 1:
  time.sleep(3600)