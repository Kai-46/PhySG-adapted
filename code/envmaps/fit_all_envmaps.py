import os
import re
import sys

sys.path.append("../..")
from multigpu_run import multigpu_run

dataset_dir = "/mnt/localssd/proj_relight/dataset_all"
job_dir = "./jobs"
os.makedirs(job_dir, exist_ok=True)

for scene in os.listdir(dataset_dir):
    scene_dir = os.path.join(dataset_dir, scene, "test")
    for envmap in os.listdir(scene_dir):
        m = re.match(r"gt_env_(\d+).hdr", envmap)
        if m is None:
            continue

        cmd = [
            "python fit_envmap_with_sg.py",
            f"{os.path.join(scene_dir, envmap)}",
            f">{os.path.join(job_dir, f'{scene}_{envmap}.log')} 2>&1",
        ]
        with open(os.path.join(job_dir, f"{scene}_{envmap}.sh"), "w") as f:
            f.write(" ".join(cmd) + "\n")


multigpu_run(job_dir, num_jobs_per_gpu=10, gpu_pool=None)
