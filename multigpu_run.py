import os
import queue
import subprocess
import threading
import time
from collections import OrderedDict


def get_gpu_count():
    std_out, std_err = subprocess.Popen(
        "nvidia-smi -L | wc -l",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).communicate()
    std_out = std_out.decode("utf-8")
    gpu_count = int(std_out.strip())
    return gpu_count


def multigpu_run(job_dir, num_jobs_per_gpu=1, gpu_pool=None):
    # Set up the job queue
    job_queue = queue.Queue()
    for job_script in sorted(os.listdir(job_dir)):
        if job_script.endswith(".sh"):
            job_queue.put(os.path.join(job_dir, job_script))
    print(f"Total {job_queue.qsize()} jobs to run")

    # We use a thread for each GPU
    # each GPU has a dictionary of max size num_jobs_per_gpu
    total_num_gpus = get_gpu_count()
    if gpu_pool is None:
        gpu_pool = list(range(total_num_gpus))
    print(f"Total {len(gpu_pool)} GPUs available")

    lock_job = threading.Lock()

    def run_jobs_on_gpu(gpu_id, num_jobs_per_gpu):
        running_jobs = OrderedDict()
        while (
            True
        ):  # continusly monitor workload in GPU and submit job if job queue is not empty and there's space on this GPU and
            # Check if there's a space on this GPU
            while (
                len(running_jobs) >= num_jobs_per_gpu
            ):  # wait until at least one process finishes if the queue is full
                for job_script, process in list(running_jobs.items()):
                    if process.poll() is not None:
                        running_jobs.pop(job_script)
                        print(f"Finished on GPU {gpu_id}: {job_script}")
            # time.sleep(10)
            for _ in range(5000000):
                pass

            # Fetch a job from the queue
            with lock_job:
                if job_queue.empty():
                    break
                job_script = job_queue.get()

            print(f"Running on GPU {gpu_id}: {job_script}")

            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} bash {job_script}"
            # print(cmd)
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
            )  # non-blocking
            running_jobs[job_script] = process

        # wait for all jobs to finish
        while len(running_jobs) > 0:
            for job_script, process in list(running_jobs.items()):
                if process.poll() is not None:
                    running_jobs.pop(job_script)
                    print(f"Finished on GPU {gpu_id}: {job_script}")
            # time.sleep(10)
            for _ in range(5000000):
                pass

    # Start the threads to run the jobs
    threads = []
    for gpu_id in gpu_pool:
        t = threading.Thread(target=run_jobs_on_gpu, args=(gpu_id, num_jobs_per_gpu))
        threads.append(t)
        t.start()

    # Wait for all the threads to finish
    for t in threads:
        t.join()

    print("All jobs finished.")
