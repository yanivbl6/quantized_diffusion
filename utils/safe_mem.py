import time

class GuardMemOp:
    def __init__(self, delay=1, retries=5):
        self.delay = delay
        self.retries = retries

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:  # An exception occurred

            cur_delay = self.delay
            for i in range(self.retries):
                try:
                    time.sleep(cur_delay*i)
                    return False  # Re-raise the exception and retry the block
                except Exception as e:
                    print(f"Retry {i+1}/{self.retries} failed: {e}")
                    if i < self.retries - 1:  # No delay on the last attempt
                        time.sleep(cur_delay*i)
            return True  # Suppress the exception if all retries failed
        return False  # No exception occurred, don't suppress