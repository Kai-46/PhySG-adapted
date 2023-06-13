import glob
import json
import os

import cv2
import imageio
import numpy as np
import torch
import utils.general as utils
from utils import rend_util


class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(
        self,
        gamma,
        instance_dir,
        train_cameras,
        max_longside=-1,
        debug_export_path=None,
        eval=False,
        eval_idx=None,
    ):
        self.instance_dir = instance_dir
        print("Creating dataset from: ", self.instance_dir)
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.gamma = gamma
        self.train_cameras = train_cameras

        self.eval = eval
        if self.eval:
            print(
                f'debug: {os.path.join(self.instance_dir, f"gt_image_{eval_idx:04d}*.png")}'
            )
            image_paths = sorted(
                glob.glob(
                    os.path.join(self.instance_dir, f"gt_image_{eval_idx:04d}*.png")
                )
            )
            mask_paths = sorted(
                glob.glob(
                    os.path.join(self.instance_dir, f"gt_mask_{eval_idx:04d}.png")
                )
            )
            camera_paths = sorted(
                glob.glob(
                    os.path.join(self.instance_dir, f"gt_camera_{eval_idx:04d}.txt")
                )
            )
        else:
            image_paths = sorted(
                glob.glob(os.path.join(self.instance_dir, "inputs/image_*.png"))
            )
            mask_paths = sorted(
                glob.glob(os.path.join(self.instance_dir, "inputs/mask_binary_*.png"))
            )
            camera_paths = sorted(
                glob.glob(os.path.join(self.instance_dir, "inputs/camera_*.txt"))
            )
        print(
            "Found # images, # masks, # cameras: ",
            len(image_paths),
            len(mask_paths),
            len(camera_paths),
        )

        bbx = np.loadtxt(
            os.path.join(self.instance_dir, "inputs/object_bounding_box.txt")
        ).reshape((3, 2))
        print("Loaded bounding box: ", bbx)
        bbx_radius = np.max(np.linalg.norm(bbx, axis=0))
        self.scene_scale = 0.85 / bbx_radius
        print("Current scene box radius: ", bbx_radius)
        print("Will scale scene unit by: ", self.scene_scale)

        def load_cam(fpath, scene_scale=1.0, img_scale=1.0):
            params = np.loadtxt(fpath)
            K, R, t, (width, height, channels) = (
                params[:3],
                params[3:6],
                params[6],
                params[7].astype(int),
            )
            K_4by4 = np.eye(4)
            K_4by4[:3, :3] = K
            W2C = np.eye(4)
            W2C[:3, :3] = R
            W2C[:3, 3] = t
            C2W = np.linalg.inv(W2C)
            if scene_scale != 1.0:
                C2W[:3, 3] *= scene_scale
                W2C = np.linalg.inv(C2W)
            if img_scale != 1.0:
                K_4by4[:2, :3] *= img_scale
            return K_4by4, C2W, width, height

        self.single_imgname = None
        self.single_imgname_idx = None
        self.sampling_idx = None

        self.n_cameras = len(image_paths)
        self.image_paths = image_paths

        self.intrinsics_all = []
        self.pose_all = []
        imgsize_all = []
        for idx in range(self.n_cameras):
            K, C2W, width, height = load_cam(camera_paths[idx], self.scene_scale)
            self.intrinsics_all.append(torch.from_numpy(K).float())
            self.pose_all.append(torch.from_numpy(C2W).float())
            imgsize_all.append([height, width])
        assert (
            np.sum(np.var(np.array(imgsize_all), axis=0)) == 0
        ), "images must be of the same resolution"
        self.img_res = imgsize_all[0]

        cur_longside = np.max(imgsize_all[0])
        if max_longside > 0 and cur_longside > max_longside:
            print(
                "Current long side is: {}, will scaled to match specified: {}".format(
                    cur_longside, max_longside
                )
            )
            img_scale = max_longside / cur_longside
            self.img_res = [int(np.round(x * img_scale)) for x in self.img_res]
            print(
                "Old img size:{}x{}, New img size: {}x{}".format(
                    imgsize_all[0][0],
                    imgsize_all[0][1],
                    self.img_res[0],
                    self.img_res[1],
                )
            )
            for idx in range(self.n_cameras):
                self.intrinsics_all[idx][0, :3] *= self.img_res[1] / imgsize_all[0][1]
                self.intrinsics_all[idx][1, :3] *= self.img_res[0] / imgsize_all[0][0]

        self.total_pixels = self.img_res[0] * self.img_res[1]

        if len(image_paths) > 0:
            assert len(image_paths) == self.n_cameras
            self.has_groundtruth = True
            self.rgb_images = []
            print("Applying inverse gamma correction: ", self.gamma)
            for idx in range(self.n_cameras):
                rgb = rend_util.load_rgb(image_paths[idx], trgt_HW=self.img_res)
                rgb = np.power(rgb, self.gamma)

                # H, W = rgb.shape[1:3]
                # assert (
                #     H == imgsize_all[idx][0] and W == imgsize_all[idx][1]
                # ), "loaded image resolution must agree with camera file"

                rgb = rgb.reshape(3, -1).transpose(1, 0)
                self.rgb_images.append(torch.from_numpy(rgb).float())
        else:
            self.has_groundtruth = False
            self.rgb_images = [
                torch.ones((self.total_pixels, 3), dtype=torch.float32),
            ] * self.n_cameras

        if len(mask_paths) > 0:
            assert len(mask_paths) == self.n_cameras
            self.object_masks = []
            for path in mask_paths:
                object_mask = rend_util.load_mask(path, trgt_HW=self.img_res)
                object_mask = object_mask.reshape(-1)
                self.object_masks.append(torch.from_numpy(object_mask).bool())
        else:
            self.object_masks = [
                torch.ones((self.total_pixels,)).bool(),
            ] * self.n_cameras

        if debug_export_path is not None:
            os.makedirs(debug_export_path, exist_ok=True)
            os.makedirs(os.path.join(debug_export_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(debug_export_path, "masks"), exist_ok=True)
            cam_dict = {}
            for idx in range(self.n_cameras):
                img_name = os.path.basename(image_paths[idx])
                cam_dict[img_name] = {
                    "img_size": [self.img_res[1], self.img_res[0]],
                    "K": self.intrinsics_all[idx]
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    .flatten()
                    .tolist(),
                    "W2C": torch.inverse(self.pose_all[idx])
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    .flatten()
                    .tolist(),
                }

                img = (
                    self.rgb_images[idx]
                    .reshape(self.img_res[0], self.img_res[1], -1)
                    .cpu()
                    .numpy()
                )
                img = np.power(img, 1.0 / self.gamma)
                imageio.imwrite(
                    os.path.join(debug_export_path, "images", img_name),
                    np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8),
                )

                mask = (
                    self.object_masks[idx]
                    .reshape(self.img_res[0], self.img_res[1])
                    .cpu()
                    .numpy()
                )
                imageio.imwrite(
                    os.path.join(debug_export_path, "masks", img_name),
                    np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8),
                )
            with open(os.path.join(debug_export_path, "cam_dict_norm.json"), "w") as fp:
                json.dump(cam_dict, fp, indent=2, sort_keys=True)

    def __len__(self):
        return self.n_cameras

    def return_single_img(self, img_name):
        self.single_imgname = img_name
        for idx in range(len(self.image_paths)):
            if os.path.basename(self.image_paths[idx]) == self.single_imgname:
                self.single_imgname_idx = idx
                break
        print("Always return: ", self.single_imgname, self.single_imgname_idx)

    def __getitem__(self, idx):
        if self.single_imgname_idx is not None:
            idx = self.single_imgname_idx

        uv = np.mgrid[0 : self.img_res[0], 0 : self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
        }

        ground_truth = {"rgb": self.rgb_images[idx]}

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def change_sampling_idx_patch(self, N_patch, r_patch=1):
        """
        :param N_patch: number of patches to be sampled
        :param r_patch: patch size will be (2*r_patch)*(2*r_patch)
        :return:
        """
        if N_patch == -1:
            self.sampling_idx = None
        else:
            # offsets to center pixels
            H, W = self.img_res
            u, v = np.meshgrid(
                np.arange(-r_patch, r_patch), np.arange(-r_patch, r_patch)
            )
            u = u.reshape(-1)
            v = v.reshape(-1)
            offsets = v * W + u
            # center pixel coordinates
            u, v = np.meshgrid(
                np.arange(r_patch, W - r_patch), np.arange(r_patch, H - r_patch)
            )
            u = u.reshape(-1)
            v = v.reshape(-1)
            select_inds = np.random.choice(u.shape[0], size=(N_patch,), replace=False)
            # convert back to original image
            select_inds = v[select_inds] * W + u[select_inds]
            # pick patches
            select_inds = np.stack([select_inds + shift for shift in offsets], axis=1)
            select_inds = select_inds.reshape(-1)
            self.sampling_idx = torch.from_numpy(select_inds).long()

    def get_pose_init(self):
        init_pose = torch.cat(
            [pose.clone().float().unsqueeze(0) for pose in self.pose_all], 0
        ).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat
