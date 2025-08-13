import os
import sys
import datetime

def setup_training_log_folder(base_path="data/logs"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(base_path, f"training_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Redirect print output
    log_file = open(os.path.join(log_dir, "console.log"), "a")
    sys.stdout = sys.stderr = Tee(sys.stdout, log_file)

    return log_dir

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()
