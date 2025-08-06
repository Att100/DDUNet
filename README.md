## DDUNet: Dual Dynamic U-Net for Highly-Efficient Cloud Segmentation

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript: 

> Yijie Li, Hewei Wang, Jinfeng Xu, Puzhen Wu, Yunzhong Xiao, Shaofan Wang, and Soumyabrata Dev, DDUNet: Dual Dynamic U-Net for Highly-Efficient Cloud Segmentation, *IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*, 2025.

### Citing DDUNet
If you find DDUNet useful in your research, please consider citing our [paper](https://arxiv.org/abs/2501.15385).
```
@article{li2025ddunet,
  title={Ddunet: Dual dynamic u-net for highly-efficient cloud segmentation},
  author={Li, Yijie and Wang, Hewei and Xu, Jinfeng and Wu, Puzhen and Xiao, Yunzhong and Wang, Shaofan and Dev, Soumyabrata},
  journal={arXiv preprint arXiv:2501.15385},
  year={2025}
}
```

## 1. Summary

Cloud segmentation identifies cloud pixels in an image, but current learning-based methods face three key challenges: (a) limited receptive field from fixed convolutional kernel sizes, (b) lack of robustness in diverse scenarios, and (c) high parameter requirements limiting real-time use. To overcome these issues, we propose the Dual Dynamic U-Net (DDUNet), a lightweight supervised model based on the U-Net architecture. DDUNet integrates dynamic multi-scale convolution (DMSC) to enhance feature merging across receptive fields and a dynamic weights and bias generator (DWBG) to improve generalization. With depth-wise convolution, DDUNet achieves 95.3% accuracy on the SWINySEG dataset using just 0.33M parameters, outperforming competing methods in both accuracy and efficiency.

![](./assets/ddunet-pipeline.svg)

## 2. Dedpendencies

### 2.1 PaddlePaddle

For CUDA 12

```
python -m pip install paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
```

### 2.2 Others

```
pip install numpy pillow pandas matplotlib scikit-learn 
```

### 2.3 Dataset

Please download the `SWINySEG` dataset from [link](https://vintage.winklerbros.net/swinyseg.html), and place the dataset in `./dataset` following the structure:

```
.
├── README.md
└── SWINySEG
    ├── GTmaps
    ├── images
    ├── metadata.csv
    ├── README.pdf

```

and then place the `txt` files in `./dataset/SWINySEG-split` to `./dataset/SWINySEG`

## 3. Usage

### 3.1 Train

`train.py` usage
```
usage: train.py [-h] [--config CONFIG] [--base_channels BASE_CHANNELS] [--batch_size BATCH_SIZE] [--lr LR] [--epochs EPOCHS] [-iou] [--dataset_split DATASET_SPLIT]
                [--dataset_path DATASET_PATH] [--eval_interval EVAL_INTERVAL]

options:
  -h, --help            show this help message and exit
  --config CONFIG       the config of model (default: full)
  --base_channels BASE_CHANNELS
                        the base_channels value of model (default: 8)
  --batch_size BATCH_SIZE
                        batchsize for model training (default: 16)
  --lr LR               the learning rate for training (default: 5e-4)
  --epochs EPOCHS       number of training epochs (default: 100)
  -iou                  use iou loss (default: False)
  --dataset_split DATASET_SPLIT
                        split of SWINySEG dataset, ['all', 'd', 'n'] (default: all)
  --dataset_path DATASET_PATH
                        path of training dataset (default: ./dataset/SWINySEG)
  --eval_interval EVAL_INTERVAL
                        interval of model evaluation during training (default: 5)
```

**Full Model**

```
python train.py
```

**Baseline+DMSC**

```
python train.py --config 'dmsc'
```

**Baseline**

```
python train.py --config 'baseline'
```

### 3.2 Test

```
python test.py
```

## 4. Results

### 4.1 Checkpoints

You can download our pre-trained checkpoints from [link](https://drive.google.com/drive/folders/1QayN1JY4SCkkT30h7lgRiLtbQVIZ0hth?usp=sharing) and place the files in `./weights`

### 4.2 Qualitative Results
![](./results/test_pred.svg)
