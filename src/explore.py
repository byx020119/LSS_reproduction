import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pyquaternion import Quaternion
from PIL import Image
from .data import compile_data
from .models import compile_model
from .public import SimpleLoss, get_val_info, gen_dx_bx, get_local_map, get_nusc_maps

"""
eval_model_iou
"""


def eval_model_iou(version,
                modelf,
                dataroot='/data/nuscenes',
                map_folder='/data/nuscenes/mini',
                gpuid=1,

                outC=1, H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=4,
                nworkers=10,
                ):

    # 传入参数
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': 5,
                    'outC': outC,
                }
    nusc_maps = get_nusc_maps(map_folder)
    trainloader, valloader = compile_data(version, dataroot, nusc_maps=nusc_maps, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=outC)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    loss_fn = SimpleLoss(1.0).cuda(gpuid)

    model.eval()
    val_info = get_val_info(model, valloader, loss_fn, device, outC=outC)
    print(val_info)


"""
viz_model_preds
"""


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
        NormalizeInverse(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        torchvision.transforms.ToPILImage(),
    ))


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)
    # plot the map
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 1], pts[:, 0], c=(0.31, 1.00, 0.50), alpha=0.2)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(1.0, 0.0, 0.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], c=(1.0, 0.0, 0.0))


def viz_model_preds(version,
                    modelf,
                    dataroot='/data/nuscenes',
                    map_folder='/data/nuscenes/mini',
                    gpuid=1,
                    viz_train=False,

                    outC=1,
                    H=900, W=1600,
                    resize_lim=(0.193, 0.225),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    rand_flip=True,

                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[4.0, 45.0, 1.0],

                    bsz=4,
                    nworkers=10,
                    ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 5,
                    'outC': outC,
                }
    nusc_maps = get_nusc_maps(map_folder)

    trainloader, valloader = compile_data(version, dataroot, nusc_maps=nusc_maps, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')
    loader = trainloader if viz_train else valloader

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=outC)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val*2, (1.5*fW + 2*fH)*val*2))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
            out = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            out = out.sigmoid().cpu()

            """
            when outC=1, out.shape=(bsz, 1, 200, 200), save pictures
            """
            if outC == 1:
                for si in range(imgs.shape[0]):
                    plt.clf()
                    for imgi, img in enumerate(imgs[si]):
                        ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                        showimg = denormalize_img(img)
                        # flip the bottom images
                        if imgi > 2:
                            showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                        plt.imshow(showimg)
                        plt.axis('off')
                        plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction', color='red')

                    ############################
                    # plot output
                    ############################
                    ax = plt.subplot(gs[0, 0])
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    plt.setp(ax.spines.values(), color='b', linewidth=2)
                    plt.legend(handles=[
                        mpatches.Patch(color=(138./255, 43./255, 226./255, 1.0), label='Vehicle Segmentation (predict)'),
                        # for visualization purposes only
                        mpatches.Patch(color=(1.0, 0.0, 0.0), label='Ego Vehicle'),
                        mpatches.Patch(color=(0.31, 1.00, 0.50, 0.5), label='Map'),
                        mlines.Line2D([], [], color=(1.0, 0.0, 0.0), alpha=0.5, label='Road divider'),
                        mlines.Line2D([], [], color=(0.0, 0.0, 1.0), alpha=0.5, label='Lane divider')
                    ], loc=(0.01, 0.80))
                    # removing dimensions of size 1 from the first dimension (indexed from 0) of the out tensor and plot
                    plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Purples')

                    # plot static map (improves visualization)
                    rec = loader.dataset.ixes[counter]
                    plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                    plt.xlim((out.shape[3], 0))
                    plt.ylim((0, out.shape[3]))
                    add_ego(bx, dx)

                    ############################
                    # plot ground truth
                    ############################
                    ax1 = plt.subplot(gs[0, 1])
                    ax1.get_xaxis().set_ticks([])
                    ax1.get_yaxis().set_ticks([])
                    plt.setp(ax1.spines.values(), color='b', linewidth=2)
                    plt.legend(handles=[
                        mpatches.Patch(color=(138./255, 43./255, 226./255, 1.0), label='Bin Imgs (ground truth)'),
                        # for visualization purposes only
                        mpatches.Patch(color=(1.0, 0.0, 0.0), label='Ego Vehicle'),
                        mpatches.Patch(color=(0.31, 1.00, 0.50, 0.5), label='Map'),
                        mlines.Line2D([], [], color=(1.0, 0.0, 0.0), alpha=0.5, label='Road divider'),
                        mlines.Line2D([], [], color=(0.0, 0.0, 1.0), alpha=0.5, label='Lane divider')
                    ], loc=(0.01, 0.80))
                    plt.imshow(binimgs[si].squeeze(0), vmin=0, vmax=1, cmap='Purples')

                    # plot static map (improves visualization)
                    rec = loader.dataset.ixes[counter]
                    plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                    plt.xlim((binimgs.shape[3], 0))
                    plt.ylim((0, binimgs.shape[3]))
                    add_ego(bx, dx)

                    ############################
                    # plot intersection over union
                    ############################
                    non_zero_elements = torch.where((out != 0) & (binimgs != 0), out, torch.tensor(0.))
                    ax2 = plt.subplot(gs[0, 2])
                    ax2.get_xaxis().set_ticks([])
                    ax2.get_yaxis().set_ticks([])
                    plt.setp(ax2.spines.values(), color='b', linewidth=2)
                    plt.legend(handles=[
                        mpatches.Patch(color=(138./255, 43./255, 226./255, 1.0),
                                       label='Iou Area (intersection over union)'),
                        # for visualization purposes only
                        mpatches.Patch(color=(1.0, 0.0, 0.0), label='Ego Vehicle'),
                        mpatches.Patch(color=(0.31, 1.00, 0.50, 0.5), label='Map'),
                        mlines.Line2D([], [], color=(1.0, 0.0, 0.0), alpha=0.5, label='Road divider'),
                        mlines.Line2D([], [], color=(0.0, 0.0, 1.0), alpha=0.5, label='Lane divider')
                    ], loc=(0.01, 0.80))
                    plt.imshow(non_zero_elements[si].squeeze(0), vmin=0, vmax=1, cmap='Purples')

                    # plot static map (improves visualization)
                    rec = loader.dataset.ixes[counter]
                    plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                    plt.xlim((non_zero_elements.shape[3], 0))
                    plt.ylim((0, non_zero_elements.shape[3]))
                    add_ego(bx, dx)

                    # # show the plot
                    # plt.show()

                    save_dir = 'runs/imgs'
                    os.makedirs(save_dir, exist_ok=True)

                    imname = os.path.join(save_dir, f'eval{batchi:06}_{si:03}.jpg')
                    print('saving', imname)
                    plt.savefig(imname)

                    counter += 1

            """
            when outC=2, out.shape=(bsz, 2, 200, 200), save pictures
            """
            if outC == 2:
                for si in range(imgs.shape[0]):
                    plt.clf()
                    for imgi, img in enumerate(imgs[si]):
                        ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                        showimg = denormalize_img(img)
                        # flip the bottom images
                        if imgi > 2:
                            showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                        plt.imshow(showimg)
                        plt.axis('off')
                        plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction', color='red')

                    ############################
                    # plot output1
                    ############################
                    ax = plt.subplot(gs[0, 0])
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    plt.setp(ax.spines.values(), color='b', linewidth=2)
                    plt.legend(handles=[
                        mpatches.Patch(color=(138. / 255, 43. / 255, 226. / 255, 1.0),
                                       label='Vehicle Segmentation (predict)'),
                        # for visualization purposes only
                        mpatches.Patch(color=(1.0, 0.0, 0.0), label='Ego Vehicle'),
                        mpatches.Patch(color=(0.31, 1.00, 0.50, 0.5), label='Map'),
                        mlines.Line2D([], [], color=(1.0, 0.0, 0.0), alpha=0.5, label='Road divider'),
                        mlines.Line2D([], [], color=(0.0, 0.0, 1.0), alpha=0.5, label='Lane divider')
                    ], loc=(0.01, 0.80))
                    plt.imshow(out[si][0].squeeze(0), vmin=0, vmax=1, cmap='Purples')

                    # plot static map (improves visualization)
                    rec = loader.dataset.ixes[counter]
                    plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                    temp = out.shape[3]
                    plt.xlim((out.shape[3], 0))
                    plt.ylim((0, out.shape[3]))
                    add_ego(bx, dx)

                    ############################
                    # plot binimgs
                    ############################
                    ax2 = plt.subplot(gs[0, 1])
                    ax2.get_xaxis().set_ticks([])
                    ax2.get_yaxis().set_ticks([])
                    plt.setp(ax2.spines.values(), color='b', linewidth=2)
                    plt.legend(handles=[
                        mpatches.Patch(color=(138. / 255, 43. / 255, 226. / 255, 1.0),
                                       label='Vehicle Segmentation (binimg)'),
                        # mpatches.Patch(color=(1, 128. / 255, 0, 1.0),
                        #                label='human Segmentation (binimg)'),
                        mpatches.Patch(color=(1.0, 0.0, 0.0), label='Ego Vehicle'),
                        mpatches.Patch(color=(0.31, 1.00, 0.50, 0.5), label='Map'),
                        mlines.Line2D([], [], color=(1.0, 0.0, 0.0), alpha=0.5, label='Road divider'),
                        mlines.Line2D([], [], color=(0.0, 0.0, 1.0), alpha=0.5, label='Lane divider')
                    ], loc=(0.01, 0.80))
                    plt.imshow(binimgs[si][0].squeeze(0), vmin=0, vmax=1, cmap='Purples')
                    # plt.imshow(binimgs[si][1].squeeze(0), alpha=0.5, vmin=0, vmax=1, cmap='Oranges')

                    # plot static map (improves visualization)
                    rec = loader.dataset.ixes[counter]
                    plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                    plt.xlim((binimgs.shape[3], 0))
                    plt.ylim((0, binimgs.shape[3]))
                    add_ego(bx, dx)

                    ############################
                    # plot output2
                    ############################
                    ax = plt.subplot(gs[0, 2])
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    plt.setp(ax.spines.values(), color='b', linewidth=2)
                    plt.legend(handles=[
                        mpatches.Patch(color=(138. / 255, 43. / 255, 226. / 255, 1.0),
                                       label='Road Segmentation (predict)'),
                        # for visualization purposes only
                        mpatches.Patch(color=(1.0, 0.0, 0.0), label='Ego Vehicle'),
                        mpatches.Patch(color=(0.31, 1.00, 0.50, 0.5), label='Map'),
                        mlines.Line2D([], [], color=(1.0, 0.0, 0.0), alpha=0.5, label='Road divider'),
                        mlines.Line2D([], [], color=(0.0, 0.0, 1.0), alpha=0.5, label='Lane divider')
                    ], loc=(0.01, 0.80))
                    plt.imshow(out[si][1].squeeze(0), vmin=0, vmax=1, cmap='Purples')

                    # plot static map (improves visualization)
                    rec = loader.dataset.ixes[counter]
                    plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                    plt.xlim((out.shape[3], 0))
                    plt.ylim((0, out.shape[3]))
                    add_ego(bx, dx)

                    # # show the plot
                    # plt.show()

                    save_dir = 'runs/imgs'
                    os.makedirs(save_dir, exist_ok=True)

                    imname = os.path.join(save_dir, f'eval{batchi:06}_{si:03}.jpg')
                    print('saving', imname)
                    plt.savefig(imname)

                    counter += 1
            """
            when outC=3, out.shape=(bsz, 3, 200, 200), save pictures
            """
            if outC == 3:
                for si in range(imgs.shape[0]):
                    plt.clf()
                    for imgi, img in enumerate(imgs[si]):
                        ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                        showimg = denormalize_img(img)
                        # flip the bottom images
                        if imgi > 2:
                            showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                        plt.imshow(showimg)
                        plt.axis('off')
                        plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction', color='red')

                    ############################
                    # plot output1
                    ############################
                    ax = plt.subplot(gs[0, 0])
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    plt.setp(ax.spines.values(), color='b', linewidth=2)
                    plt.legend(handles=[
                        mpatches.Patch(color=(138. / 255, 43. / 255, 226. / 255, 1.0),
                                       label='Vehicle Segmentation (predict)'),
                        # for visualization purposes only
                        mpatches.Patch(color=(1.0, 0.0, 0.0), label='Ego Vehicle'),
                        mpatches.Patch(color=(0.31, 1.00, 0.50, 0.5), label='Map'),
                        mlines.Line2D([], [], color=(1.0, 0.0, 0.0), alpha=0.5, label='Road divider'),
                        mlines.Line2D([], [], color=(0.0, 0.0, 1.0), alpha=0.5, label='Lane divider')
                    ], loc=(0.01, 0.80))
                    plt.imshow(out[si][0].squeeze(0), vmin=0, vmax=1, cmap='Purples')

                    # plot static map (improves visualization)
                    rec = loader.dataset.ixes[counter]
                    plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                    temp = out.shape[3]
                    plt.xlim((out.shape[3], 0))
                    plt.ylim((0, out.shape[3]))
                    add_ego(bx, dx)

                    ############################
                    # plot output2
                    ############################
                    ax2 = plt.subplot(gs[0, 1])
                    ax2.get_xaxis().set_ticks([])
                    ax2.get_yaxis().set_ticks([])
                    plt.setp(ax2.spines.values(), color='b', linewidth=2)
                    plt.legend(handles=[
                        mpatches.Patch(color=(138. / 255, 43. / 255, 226. / 255, 1.0),
                                       label='Divider Segmentation (predict)'),
                        mpatches.Patch(color=(1.0, 0.0, 0.0), label='Ego Vehicle'),
                        mpatches.Patch(color=(0.31, 1.00, 0.50, 0.5), label='Map'),
                        mlines.Line2D([], [], color=(1.0, 0.0, 0.0), alpha=0.5, label='Road divider'),
                        mlines.Line2D([], [], color=(0.0, 0.0, 1.0), alpha=0.5, label='Lane divider')
                    ], loc=(0.01, 0.80))
                    plt.imshow(out[si][1].squeeze(0), vmin=0, vmax=1, cmap='Purples')

                    # plot static map (improves visualization)
                    rec = loader.dataset.ixes[counter]
                    plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                    plt.xlim((out.shape[3], 0))
                    plt.ylim((0, out.shape[3]))
                    add_ego(bx, dx)

                    ############################
                    # plot output3
                    ############################
                    ax = plt.subplot(gs[0, 2])
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    plt.setp(ax.spines.values(), color='b', linewidth=2)
                    plt.legend(handles=[
                        mpatches.Patch(color=(138. / 255, 43. / 255, 226. / 255, 1.0),
                                       label='Road Segmentation (predict)'),
                        # for visualization purposes only
                        mpatches.Patch(color=(1.0, 0.0, 0.0), label='Ego Vehicle'),
                        mpatches.Patch(color=(0.31, 1.00, 0.50, 0.5), label='Map'),
                        mlines.Line2D([], [], color=(1.0, 0.0, 0.0), alpha=0.5, label='Road divider'),
                        mlines.Line2D([], [], color=(0.0, 0.0, 1.0), alpha=0.5, label='Lane divider')
                    ], loc=(0.01, 0.80))
                    plt.imshow(out[si][2].squeeze(0), vmin=0, vmax=1, cmap='Purples')

                    # plot static map (improves visualization)
                    rec = loader.dataset.ixes[counter]
                    plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                    plt.xlim((out.shape[3], 0))
                    plt.ylim((0, out.shape[3]))
                    add_ego(bx, dx)

                    # # show the plot
                    # plt.show()

                    save_dir = 'runs/imgs'
                    os.makedirs(save_dir, exist_ok=True)

                    imname = os.path.join(save_dir, f'eval{batchi:06}_{si:03}.jpg')
                    print('saving', imname)
                    plt.savefig(imname)

                    counter += 1





