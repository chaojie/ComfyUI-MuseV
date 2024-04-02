from multiprocessing import Pool, Process, Value, Lock, Pool
from threading import Thread
from multiprocessing.pool import ThreadPool
from typing import List, Callable, Any, Tuple
import multiprocessing as mp


from functools import partial


def run_task_in_parallel(
    worker: Callable,
    tasks: List[Any],
    num: int = 1,
    process_type: str = "process",
    use_counter: bool = False,
    print_freq: int = 100,
) -> Tuple[List, None]:
    """并行运行任务的设置

    Args:
        worker (func): 工作函数
        tasks (iterator): 任务队列
        num (int, optional): 并行处理核数量. Defaults to 1.
        process_type (str, optional): 并行处理类型，process or thread. Defaults to "process".
        use_counter (bool, optional): 是否使用全局计数器. Defaults to False.
        print_freq (int, optional): 打印计算进展. Defaults to 100.q
    """
    results = []
    n_task = len(tasks)
    if num == 1:
        for i, task in enumerate(tasks):
            result = worker(task)
            results.append(result)
            if use_counter:
                if i % print_freq == 0:
                    print(f"finished {n_task}/{i} tasks")
    else:
        if use_counter:
            raise NotImplementedError(
                "not supported counter for multiprocess/multithread"
            )
        process_cls = Pool if process_type == "process" else ThreadPool
        with process_cls(num) as p:
            results = p.map(worker, tasks)
    return results


def prepare_task(tasks, queue, lock):
    for i, task in enumerate(tasks):
        with lock:
            if i % 100 == 0:
                # if i % 1 == 0:
                print("prepare task: ", i, task)
            queue.put(task)


def worker_task(worker, queue, lock, counter):
    while True:
        with lock:
            task = queue.get()
            counter.value += 1
            print("finish task: ", counter.value, task)
        if task is None:
            break
        worker.do_task(task)


def run_pipeline(worker_class, tasks, n_process=4):
    # lock = mp.Lock()
    # 用不上却会影响死锁？
    manager = mp.Manager()
    # lock = manager.Lock()
    # queue = manager.Queue()
    # counter = manager.Value("i", 0)
    lock = mp.Lock()
    queue = mp.Queue()
    counter = mp.Value("i", 0)
    prepare_process = mp.Process(target=prepare_task, args=(tasks, queue, lock))
    prepare_process.start()
    worker_processes = []
    for i in range(n_process):  # number of worker processes
        worker = worker_class()
        worker_process = mp.Process(
            target=worker_task, args=(worker, queue, lock, counter)
        )
        worker_process.start()
        worker_processes.append(worker_process)

    prepare_process.join()
    # queue.join()

    for i in range(n_process):  # number of worker processes
        queue.put(None)

    for worker_process in worker_processes:
        worker_process.join()
