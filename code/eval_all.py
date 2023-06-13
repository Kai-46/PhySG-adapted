import os
import re
import sys

sys.path.append("..")
from multigpu_run import multigpu_run

dataset_dir = "/mnt/localssd/proj_relight/dataset_all"
job_dir = "./jobs_eval"
os.makedirs(job_dir, exist_ok=True)

for scene in os.listdir(dataset_dir):
    scene_dir = os.path.join(dataset_dir, scene, "test")

    for envmap in os.listdir(scene_dir):
        m = re.match(r"gt_env_(\d+).hdr", envmap)
        if m is None:
            continue

        cmd = [
            "python evaluation/eval.py --conf confs_sg/default.conf",
            f"--data_split_dir {scene_dir}",
            f"--expname {scene}",
            f"--resolution 256 --save_exr --gamma 2.2",
            f"--light_sg {os.path.join(scene_dir, envmap[:-4], 'sg_128.npy')}",
            f">{os.path.join(job_dir, f'{scene}_{envmap}.log')} 2>&1",
        ]

        def get_nice_cmd_str(cmd):
            return " \\\n    ".join(cmd)

        with open(os.path.join(job_dir, f"{scene}_{envmap}.sh"), "w") as f:
            f.write(get_nice_cmd_str(cmd) + "\n")


multigpu_run(job_dir, num_jobs_per_gpu=1, gpu_pool=None)

# gather final results
base_eval_dir = "../evals"
gather_dir = os.path.join(base_eval_dir, "gather")
os.makedirs(gather_dir, exist_ok=True)

for scene in os.listdir(base_eval_dir):
    if scene.find("default-") == -1:
        continue
    scene_name = scene[len("default-") :]
    gather_scene_dir = os.path.join(gather_dir, scene_name)
    os.makedirs(gather_scene_dir, exist_ok=True)
    os.system(f"cp {os.path.join(base_eval_dir, scene, '**/*.exr')} {gather_scene_dir}")
