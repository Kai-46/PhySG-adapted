import os
import sys

sys.path.append("..")
from multigpu_run import multigpu_run


dataset_dir = "/mnt/localssd/proj_relight/dataset_all"
job_dir = "./jobs"
os.makedirs(job_dir, exist_ok=True)

for scene in os.listdir(dataset_dir):
    scene_dir = os.path.join(dataset_dir, scene, "test")

    cmd = [
        "python training/exp_runner.py --conf confs_sg/default.conf",
        f"--data_split_dir {scene_dir}",
        f"--expname {scene}",
        f"--nepoch 4000 --max_niter 400001 --gamma 2.2",
        f">{os.path.join(job_dir, f'{scene}.log')} 2>&1",
    ]

    def get_nice_cmd_str(cmd):
        return " \\\n    ".join(cmd)

    with open(os.path.join(job_dir, f"{scene}.sh"), "w") as f:
        f.write(get_nice_cmd_str(cmd) + "\n")


multigpu_run(job_dir, num_jobs_per_gpu=1, gpu_pool=None)
