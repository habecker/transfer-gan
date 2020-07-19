from typing import Callable, Any
import multiprocessing

class ParallelProcessing:
    def __init__(self, num_processes:int, initializer:Callable, initargs:tuple, task:Callable, job:list, do_result:Callable):
        self.pool = multiprocessing.Pool(processes=num_processes, initializer = initializer, initargs = initargs)
        self.task = task
        self.do_result = do_result
        self.job = job
        self.completed = False

    def start(self):
        if self.completed:
            raise Exception('Job already done')
        # Start a Pool with 8 processes
        for res in self.pool.imap_unordered(self.task, self.job):
            self.do_result(res)
        # Safely terminate the pool
        self.pool.close()
        self.pool.join()
        self.completed = True
