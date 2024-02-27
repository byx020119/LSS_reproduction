from fire import Fire
import src

"""
eg:
eval_model_iou mini --modelf=./model525000.pt --dataroot=../dataset/nuScenes  --map_folder=../dataset/nuScenes/mini --outC=1 --gpuid=0 
viz_model_preds mini --modelf=./model525000.pt --dataroot=../dataset/nuScenes --map_folder=../dataset/nuScenes/mini  --outC=1 --gpuid=0
train mini  --dataroot=../dataset/nuScenes  --pretrained_weights_path=./runs/weights/model525000 --map_folder=../dataset/nuScenes/mini --outC=1 --gpuid=0

annotations: outC=class number
    outC=1:vehicle
    outC=2:vehicle and human
    outC=3:vehicle, human and road
"""


if __name__ == '__main__':
    Fire({
        'eval_model_iou': src.explore.eval_model_iou,
        'train': src.train.train,
        'viz_model_preds': src.explore.viz_model_preds,
    })
