import logging
from logging.handlers import QueueHandler, QueueListener
import queue
import threading
import time

class BufferedLogger:
    """
    Logger that buffers messages in memeory and writes to a file in batches
    """

    def __init__(self, log_file, buffer_size=100):
        self.log_file = log_file
        self.buffer_size = buffer_size
        self.buffer = []

    def log(self, message):
        self.buffer.append(message)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
        
    def flush(self):
        if not self.buffer:
            return
        with open(self.log_file, 'a') as f:
            f.write('\n'.join(self.buffer) + '\n')
        self.buffer.clear()
    
    def close(self):
        self.flush()

class AsyncLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.log_queue = queue.Queue(-1)

        self.queue_handler = QueueHandler(self.log_queue)
        self.logger = logging.getLogger("AsyncLogger")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.queue_handler)

        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

        self.listener = QueueListener(self.log_queue, self.file_handler)
        self.listener.start()

    def log(self, message):
        self.logger.info(message)

    def close(self):
        self.listener.stop()
