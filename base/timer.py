# timer.py

import time
MIN_LOG_COUNT = 10


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self, name: str = 'Timer', logger=None):
        self.name = name
        self.logger = logger
        self._start_time = time.perf_counter()

    def start(self, reset: bool = True):
        """Start a new timer"""
        if not reset and self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self, count: int = None):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        if count is not None:
            if count < MIN_LOG_COUNT:
                return elapsed_time
            speed = count//elapsed_time
            if speed > 1000000:
                speed = f"{speed/1000000:.2f}M"
            elif speed > 1000:
                speed = f"{speed/1000:.2f}k"
        else:
            speed = 'N/A'
        self._start_time = None
        if self.logger:
            self.logger.info(
                f"{self.name} took {elapsed_time:0.4f} seconds {speed}/sec {count or 'N/A'} count")
        else:
            print(
                f"{self.name} Elapsed time: {elapsed_time:0.4f} s {speed}/sec {count or 'N/A'} count")
        return elapsed_time
