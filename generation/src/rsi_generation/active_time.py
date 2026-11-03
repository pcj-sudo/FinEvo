"""Persist running time independently of task creation and queue timestamps."""
import threading
import time
from .files import read_json, write_json


class ActiveTime:
    def __init__(self, path):
        self.path = path
        self.prior = read_json(path, {})
        self.base = self.prior.get('elapsed_sec', 0)
        self.start = time.monotonic()
        self.stop = threading.Event()
        self.thread = None

    def elapsed(self):
        return self.base + time.monotonic() - self.start

    def save(self, running=True):
        write_json(self.path, {**self.prior, 'elapsed_sec': self.elapsed(),
                              'updated_at': time.time(), 'running': running})

    def __enter__(self):
        self.save()
        def heartbeat():
            while not self.stop.wait(30):
                self.save()
        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.save(False)
