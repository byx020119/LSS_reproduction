# LSS_reproduction

## Contribution

Based on the official [open-source implementation](https://nv-tlabs.github.io/lift-splat-shoot/), we have extended and augmented the code, adding enhancements specifically for vehicle, lane-line, and road segmentation components in the reproducibility of the LSS approach.And achieved slightly better results than those mentioned in the [Paper](https://arxiv.org/abs/2008.05711).

在复现LSS的基础上，在[官方开源](https://nv-tlabs.github.io/lift-splat-shoot/)的基础上对代码进行补充，完善了车辆、车道线、道路分割部分。并获得了比[论文](https://arxiv.org/abs/2008.05711)中提到的稍好的效果。


## Quikly start

```shell
git clone https://github.com/byx020119/LSS_reproduction.git

cd LSS_reproduction

python main.py viz_model_preds mini --modelf=MODEL_ROOT --dataroot=NUSCENES_ROOT --map_folder=MAP_FOLDER_ROOT  --outC=CLASS_NUMBER --gpuid=GPU_ID
```
annotations: outC=class number

default:
- outC=1:vehicle
- outC=2:vehicle and road
- outC=3:vehicle, divider and road

## Demo

![[./source/demo.gif]]

## Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D
PyTorch code for Lift-Splat-Shoot (ECCV 2020).

Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D
Jonah Philion, Sanja Fidler
ECCV, 2020 (Poster)

[Paper](https://arxiv.org/abs/2008.05711)

Abstract: The goal of perception for autonomous vehicles is to extract semantic representations from multiple sensors and fuse these representations into a single "bird's-eye-view" coordinate frame for consumption by motion planning. We propose a new end-to-end architecture that directly extracts a bird's-eye-view representation of a scene given image data from an arbitrary number of cameras. The core idea behind our approach is to "lift" each image individually into a frustum of features for each camera, then "splat" all frustums into a rasterized bird's-eye-view grid. By training on the entire camera rig, we provide evidence that our model is able to learn not only how to represent images but how to fuse predictions from all cameras into a single cohesive representation of the scene while being robust to calibration error. On standard bird's-eye-view tasks such as object segmentation and map segmentation, our model outperforms all baselines and prior work. In pursuit of the goal of learning dense representations for motion planning, we show that the representations inferred by our model enable interpretable end-to-end motion planning by "shooting" template trajectories into a bird's-eye-view cost map output by our network. We benchmark our approach against models that use oracle depth from lidar. Project page: https://nv-tlabs.github.io/lift-splat-shoot/.



## Citation
If you found this codebase useful in your research, please consider citing
```
@inproceedings{philion2020lift,
    title={Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D},
    author={Jonah Philion and Sanja Fidler},
    booktitle={Proceedings of the European Conference on Computer Vision},
    year={2020},
}
```