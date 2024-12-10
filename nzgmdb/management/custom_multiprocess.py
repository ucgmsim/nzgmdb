import multiprocessing as mp
from queue import Empty
from typing import Any, Callable


def worker(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    finished_queue: mp.Queue,
    func: Callable[..., Any],
    *args: Any,
) -> None:
    """
    Worker function to process tasks from the task queue and put results in the result queue.

    Parameters
    ----------
    task_queue : multiprocessing.Queue
        Queue from which tasks are retrieved.
    result_queue : multiprocessing.Queue
        Queue where results are put.
    finished_queue : multiprocessing.Queue
        Queue to signal that the process has finished.
    func : Callable[..., Any]
        Function to process each task.
    *args : Any
        Additional arguments to pass to the function.
    """
    while True:
        try:
            # Get the task
            task = task_queue.get_nowait()
        except Empty:
            break

        # Perform the task
        result = func(task, *args)
        result_queue.put(result)

    # Signal that the process has finished
    finished_queue.put(mp.current_process().pid)
    # Sentinel value to signal the end of results
    result_queue.put(None)


def custom_multiprocess(
    func: Callable[..., Any], tasks: list[Any], n_procs: int, *args: Any
) -> list[Any]:
    """
    Custom multiprocess function to distribute tasks among multiple processes.

    Parameters
    ----------
    func : Callable[..., Any]
        Function to process each task.
    tasks : list[Any]
        List of tasks to be processed.
    n_procs : int
        Number of processes to use.
    *args : Any
        Additional arguments to pass to the function.

    Returns
    -------
    list[Any]
        List of results from processing the tasks.
    """
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    finished_queue = mp.Queue()

    # Add tasks to the queue
    for task in tasks:
        task_queue.put(task)

    # Start the processes
    processes = {}
    for _ in range(n_procs):
        p = mp.Process(
            target=worker, args=(task_queue, result_queue, finished_queue, func, *args)
        )
        p.start()
        processes[p.pid] = p

    # Wait for all processes to finish
    processes_to_end = []
    while processes:
        pid = finished_queue.get()
        processes_to_end.append(processes.pop(pid))

    # Collect results until all sentinel values are received
    results = []
    sentinel_count = 0
    while sentinel_count < n_procs:
        result = result_queue.get()
        if result is None:
            sentinel_count += 1
        else:
            results.append(result)

    # Terminate the processes
    for p in processes_to_end:
        p.terminate()

    return results
