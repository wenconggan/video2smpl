# Video2Smpl


## Environment Setup

```bash
cd video2smpl
conda create -y -n video2smpl python=3.10
conda activate video2smpl
pip install -r requirements.txt
pip install -e .
```

## Inputs & Outputs

```bash
mkdir inputs
mkdir outputs
```


```bash
mkdir -p inputs/checkpoints

# 1. You need to sign up for downloading [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/). And the checkpoints should be placed in the following structure:

inputs/checkpoints/
├── body_models/smplx/
│   └── SMPLX_{GENDER}.npz # SMPLX (We predict SMPLX params + evaluation)
└── body_models/smpl/
    └── SMPL_{GENDER}.pkl  # SMPL (rendering and evaluation)

# 2. Download other pretrained models from Google-Drive (By downloading, you agree to the corresponding licences): https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link

inputs/checkpoints/
├── dpvo/
│   └── dpvo.pth
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt
```

### video2pt
Demo entries are provided in `tools/demo`. Use `-s` to skip visual odometry if you know the camera is static, otherwise the camera will be estimated by DPVO.
We also provide a script `demo_folder.py` to inference a entire folder.
```shell
python tools/demo/demo.py --video=example_video/tennis.mp4 -s
python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
```

### pt2npz

```shell
cd pt2npz
python Converter.py  --input + pt path
python Converter.py  --input /home/wenconggan/视频/video2smpl/outputs/demo/tennis/hmr4d_results.pt 

```

### npz retarget 

```
phc retarget / mink retarget 

