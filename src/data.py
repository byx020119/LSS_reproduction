import torch
import torchvision
import os
import numpy as np
import cv2
import matplotlib as mpl
from matplotlib.path import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from .public import gen_dx_bx, get_rot, get_local_map
from PIL import Image
from pyquaternion import Quaternion
from glob import glob

mpl.use('Agg')

"""
Nuscdata
"""


def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))


# def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
#     """
#     Returns at most nsweeps of lidar in the ego frame.
#     Returned tensor is 5(x, y, z, reflectance, dt) x N
#     Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
#     """
#     points = np.zeros((5, 0))
#
#     # Get reference pose and timestamp.
#     ref_sd_token = sample_rec['data']['LIDAR_TOP']
#     ref_sd_rec = nusc.get('sample_data', ref_sd_token)
#     ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
#     ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
#     ref_time = 1e-6 * ref_sd_rec['timestamp']
#
#     # Homogeneous transformation matrix from global to _current_ ego car frame.
#     car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
#                                         inverse=True)
#
#     # Aggregate current and previous sweeps.
#     sample_data_token = sample_rec['data']['LIDAR_TOP']
#     current_sd_rec = nusc.get('sample_data', sample_data_token)
#     for _ in range(nsweeps):
#         # Load up the pointcloud and remove points close to the sensor.
#         current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
#         current_pc.remove_close(min_distance)
#
#         # Get past pose.
#         current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
#         global_from_car = transform_matrix(current_pose_rec['translation'],
#                                             Quaternion(current_pose_rec['rotation']), inverse=False)
#
#         # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
#         current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
#         car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
#                                             inverse=False)
#
#         # Fuse four transformation matrices into one and perform transform.
#         trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
#         current_pc.transform(trans_matrix)
#
#         # Add time vector which can be used as a temporal feature.
#         time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
#         times = time_lag * np.ones((1, current_pc.nbr_points()))
#
#         new_points = np.concatenate((current_pc.points, times), 0)
#         points = np.concatenate((points, new_points), 1)
#
#         # Abort if there are no previous sweeps.
#         if current_sd_rec['prev'] == '':
#             break
#         else:
#             current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])
#
#     return points


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf, nusc_maps):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.nusc_maps = nusc_maps

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                       resize=resize,
                                                       resize_dims=resize_dims,
                                                       crop=crop,
                                                       flip=flip,
                                                       rotate=rotate,
                                                       )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    # def get_lidar_data(self, rec, nsweeps):
    #     pts = get_lidar_data(self.nusc, rec,
    #                    nsweeps=nsweeps, min_distance=2.2)
    #     return torch.Tensor(pts)[:3]  # x,y,z

    def get_binmap(self, rec, nusc_maps):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])

        scene2map = {}
        for rec_temp in self.nusc.scene:
            log = self.nusc.get('log', rec_temp['log_token'])
            scene2map[rec_temp['name']] = log['location']

        map_name = scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        poly_names = ['road_segment', 'lane']
        line_names = ['road_divider', 'lane_divider']
        lmap = get_local_map(nusc_maps[map_name], center,
                             50.0, poly_names, line_names)

        # Delta X, Base X，Number X
        dx, bx, _ = gen_dx_bx(self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound'])
        dx, bx = dx[:2].numpy(), bx[:2].numpy()
        filled_array = np.zeros((self.nx[0], self.nx[1]))

        # Loop to fill polygons
        for name in poly_names:
            for la in lmap[name]:
                # Convert vertex coordinates to grid coordinates
                pts = ((la - bx) / dx).astype(int)
                # Create a Path object for using the contains_points method
                path = Path(pts)
                # Generate grid coordinates
                xx, yy = np.meshgrid(np.arange(self.nx[1]), np.arange(self.nx[0]))
                grid_points = np.vstack((yy.ravel(), xx.ravel())).T
                # Check if each point is inside the polygon
                inside_polygon = path.contains_points(grid_points)
                # Fill the points inside the polygon with 1
                filled_array[grid_points[inside_polygon, 0], grid_points[inside_polygon, 1]] = 1

        return filled_array

    def get_binimg(self, rec):
        """
        Get a binary image of the scene.
        shape (1,200,200) class number 1
        """
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            # fillPoly takes pts in (y,x) format
            cv2.fillPoly(img, [pts], 1.0)

        # import matplotlib.pyplot as plt
        # plt.imshow(img, vmin=0, vmax=1, cmap='Purples')
        # plt.show()
        return torch.Tensor(img).unsqueeze(0)

    def get_binimg_s2(self, rec):
        """
        Get a binary image of the scene.
        shape (2,200,200) class number 2
        """
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((2, self.nx[0], self.nx[1]))  # outC = 2
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if inst['category_name'].split('.')[0] == 'vehicle':
                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(trans)
                box.rotate(rot)

                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                # fillPoly takes pts in (y,x) format
                cv2.fillPoly(img[0], [pts], 1.0)
            # add category such as: human, animal, etc.
            elif inst['category_name'].split('.')[0] == 'human':

                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(trans)
                box.rotate(rot)

                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                # fillPoly takes pts in (y,x) format
                cv2.fillPoly(img[1], [pts], 1.0)

        # import matplotlib.pyplot as plt
        # plt.imshow(img[0], vmin=0, vmax=1, cmap='Purples')
        # plt.show()
        return torch.Tensor(img)

    def get_binimg_s3(self, rec):
        """
        Get a binary image of the scene.
        shape (3,200,200) class number 3
        """
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((3, self.nx[0], self.nx[1]))  # outC = 3
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if inst['category_name'].split('.')[0] == 'vehicle':

                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(trans)
                box.rotate(rot)

                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                # fillPoly takes pts in (y,x) format
                cv2.fillPoly(img[0], [pts], 1.0)
            # add category such as: human, animal, etc.
            elif inst['category_name'].split('.')[0] == 'human':

                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(trans)
                box.rotate(rot)

                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                # fillPoly takes pts in (y,x) format
                cv2.fillPoly(img[1], [pts], 1.0)
            # add road category
            img[2] = self.get_binmap(rec, self.nusc_maps)

        # import matplotlib.pyplot as plt
        # plt.imshow(img[0], vmin=0, vmax=1, cmap='Purples')
        # plt.show()
        return torch.Tensor(img)

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


"""
viz data and segmentation data
"""


# class VizData(NuscData):
#     def __init__(self, *args, **kwargs):
#         super(VizData, self).__init__(*args, **kwargs)
#
#     def __getitem__(self, index):
#         rec = self.ixes[index]
#
#         cams = self.choose_cams()
#         imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
#         lidar_data = self.get_lidar_data(rec, nsweeps=3)
#         binimg = self.get_binimg(rec)
#
#         return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        if self.data_aug_conf['outC'] == 1:
            binimg = self.get_binimg(rec)
        elif self.data_aug_conf['outC'] == 2:
            binimg = self.get_binimg_s2(rec)
        elif self.data_aug_conf['outC'] == 3:
            binimg = self.get_binimg_s3(rec)

        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


"""
compile_data
"""


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, nusc_maps, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)
    parser = {
        # 'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]
    traindata = parser(nusc, is_train=True, nusc_maps=nusc_maps, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf)
    valdata = parser(nusc, is_train=False, nusc_maps=nusc_maps, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader

