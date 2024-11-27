import multiprocessing as mp
from queue import Empty
from typing import Any, Callable

from nzgmdb.mseed_management import creation


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
            print(f"Queue {mp.current_process().pid} Empty")
            break

        # Perform the task
        try:
            print(f"Processing task: {task.code} for {mp.current_process().pid}")
            result = func(task, *args)
            result_queue.put(result)
        except Exception as e:
            print(f"Error processing task: {task} for {mp.current_process().pid}")
            print(e)

    # Signal that the process has finished
    finished_queue.put(mp.current_process().pid)
    # Sentinel value to signal the end of results
    result_queue.put(None)
    print(f"Process {mp.current_process().pid} finished")


# Worker function that handles file-writing tasks
def file_writer_worker(queue, finish_queue):
    while True:
        task = queue.get()  # Get a task from the queue
        if task is None:  # Stop signal
            break
        mseed, event_id, station_code, mseed_dir = task
        creation.write_mseed(mseed, event_id, station_code, mseed_dir)
    finish_queue.put(mp.current_process().pid)  # Signal that the process has finished
    print(f"File writer process {mp.current_process().pid} finished")


def custom_multiprocess(
    func: Callable[..., Any],
    tasks: list[Any],
    n_procs: int,
    writing_process=False,
    *args: Any,
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
    writing_process : bool
        Whether to create a specific writing process to write files to handle IO.
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

    if writing_process:
        n_procs -= 1
        # Queue for file-writing tasks
        writing_queue = mp.Queue()
        finished_writing_queue = mp.Queue()
        # Start the file writer worker
        file_writer_process = mp.Process(
            target=file_writer_worker, args=(writing_queue, finished_writing_queue)
        )
        file_writer_process.start()

        # Add the writing queue to the arguments
        args = (writing_queue,) + args

    # Check length of tasks and reduce n_procs if necessary
    n_procs = min(n_procs, len(tasks))

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
        print(f"Process {pid} finished")
        processes_to_end.append(processes.pop(pid))

    print("Collecting Results")
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

    if writing_process:
        print("Adding stop signal to file writer queue")
        writing_queue.put(None)  # Stop the file writer process
        pid = finished_writing_queue.get()
        print("Terminating file writer process")
        file_writer_process.terminate()

    return results
