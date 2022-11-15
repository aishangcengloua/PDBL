# [PDBL: Improving Histopathological Tissue Classification with Plug-and-Play Pyramidal Deep-Broad Learning](https://ieeexplore.ieee.org/document/9740140)

## Introduction

In this paper, the authors propose a framework called PDBL for classification tasks.

![outline](https://github.com/linjiatai/PDBL/raw/main/PDBL.png)

In the paper, four models were trained on different data sets and the performance was compared. Four models are **Baseline+FC***, **Baseline+PDBL**, **Baseline***, **Baseline*+PDBL** :

- `Baseline+FC*` : Frozethe baseline models pre-trained on ImageNet and only update the fully connected layers (FC). But The epochs for training FC are not mentioned in the paper. Therefore, I train FC for 50 epochs.
- `Baseline+PDBL`  : PDBL directly plugged on the baseline models pre-trained by ImageNet.
- `Baseline*` : baseline models pre-trained by ImageNet fine-tuned for 50 epochs without PDBL.
- `Baseline*+PDBL`  : Baseline models pre-trained by ImageNet fine-tuned for 50 epochs with PDBL

## Structure

```python
../PDBL
├── checkpoint
│   ├── KMI_001
│   └── LC_001
├── PDBL_Dataset
│   ├── Kather
│   │   ├── CRC-VAL-HE-7K
│   │   ├── KMI_001
│   │   └── NCT-CRC-HE-100K
│   ├── LC25000
│   ├── LC_Test
│   ├── LC_Train_001
│   └── LC_Train_100
├── models
│   ├── efficientnet.py
│   ├── resnet.py
│   ├── shufflenet.py
│   ├── utils.py
│   └── __init__.py
├── data_split.py
├── dataset.py
├── pdbl.py
└── trainer.py
```

The role of each file or folder is as follows:

- `checkpoint` : save the parameters of models with different datasets.
- `PDBL_Dataset` : save Kather and LC25000 dataset. In Kather, CRC-VAL-HE-7K, KMI_001, NCT-CRC-HE-100K are test set, 1% training set and 100 training set samples respectively. In LC25000, LC25000, LC_Test, LC_Train_001, LC_Train_100 are the entire data set, test set, 1% training set and 100% training set samples respectively.
- `models` : architecture of the three models. The models are shufflenet, efficientnet and resnet which are used for classification tasks.
- `data_split.py` : split LC25000 dataset into 60% test set and 40% train set.
- `dataset.py` : make Dataset and DataLoader.
- `pdbl.py` : build pdbl which is used to assist the above three models to classify.
- `trainer.py` : train and test all models.

## Requirements

- numpy==1.20.1
- opencv_contrib_python==4.5.4.60
- scikit_learn==1.1.3
- torch==1.12.0
- torchvision==0.13.0
- tqdm==4.59.0

## Usage

### Installation

- Download the repository.

```python
git clone https://github.com/aishangcengloua/PDBL.git
```

- Install python dependencies.

```python
pip install -r requirements.txt
```

### Training and Inference

I provide an example to create and train a PDBL on 1% KMI set without CNN re-training burden. To use this example code, you must download the data set, you must download the 1% KMI set and KME set ([Baidu Netdisk](https://pan.baidu.com/s/1gLRDYK2lmgoLlZuzLcNIfw?pwd=wfzk) with code **wfzk**) and unpacked them in **dataset** folder. And you can train and test the PDBL by the command:

```python
python trainer.py --save_dir checkpoint/Kather_001 --train_dir PDBL_Dataset/Kather/KMI_001 --val_dir PDBL_Dataset/Kather/CRC-VAL-HE-7K --n_classes 9 --train_model False --train_fc False --fine_tuning False
```

In the command, `train_model` is used to decide whether to train the classification model and `train_fc` and `fine_tuning` are used to decide whether to train FC or fine-tune the model.

Moreover, the original Kather Dataset and LC25000 Dataset can be download at the links [Kather2019](https://zenodo.org/record/1214456) and [LC25000](https://github.com/tampapath/lung_colon_image_set).

## Reference

- [**https://github.com/linjiatai/PDBL**](https://github.com/linjiatai/PDBL)