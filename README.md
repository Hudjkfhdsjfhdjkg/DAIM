# DAIM
Implementation of our paper, "High Feature Distinguishability for Adaptive Image-text
Matching with Dual-stream Transformers". This repo is built on top of [METER](https://github.com/zdou0830/METER).

## Introduction
Recently, most image-text matching (ITM) approaches have embraced a dual-stream Transformer architecture to facilitate the learning and alignment of cross-modal semantic information. Despite the efficacy of this
methodology in bridging the semantic disparity between images and texts, it exhibits two primary limitations.
Firstly, it falls short in discriminating the nuanced similarities among features, which leads to misleading
outcomes or even compromises the overall ITM process. Secondly, the conventional triplet training paradigm
relies on a pre-determined, fixed margin coefficient, thereby impeding its capacity to accurately gauge
the similarity relationships between positive and negative samples. In this paper, we propose high feature
Distinguishability for Adaptive Image-text Matching with dual-stream transformers (termed as DAIM). To
address the first limitation, we design a feature discriminability module to bring similar features closer
together but with a certain degree of distinction and push dissimilar features farther apart, resulting in high
feature distinguishability for accurate ITM. To address the second limitation, we devise a margin optimization
module to perceive the similarity distribution between positive and negative samples in real time during
training, thereby adaptively adjusting the margin coefficient to minimize the cross-modal semantic gap to
the greatest extent possible. Based on this, we align the multi-level (i.e., representations from low-, middle-,
and high-layer transformer encoders) semantic information of cross-modal data by adaptively optimizing the
semantic distributions of positive and negative samples. We conduct extensive experiments on two commonly
used benchmark datasets, including MSCOCO and Flickr30K. Experimental results verify that DAIM can
achieve a higher performance (e.g., 4.7% RSUM gain on MSCOCO) than the state-of-the-art ITM methods.

![model](D:\download\Architecture (1).png)

## Requirements 
We recommended the following dependencies.

* Python 3.8 
* [PyTorch](http://pytorch.org/) (1.8.1)
* [NumPy](http://www.numpy.org/) (>=1.23.4)
* [transformers](https://huggingface.co/docs/transformers) (4.6.0)
* [timm](https://timm.fast.ai/) (0.4.12)
* [torchvision]()

For details, check the [requirements.txt](https://github.com/LuminosityX/HAT/blob/main/requirements.txt) file

## Download data and pretrained model

The raw images can be downloaded from their original sources [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/). We refer to the path of extracted files as `$DATA_PATH`.

If you don't want to train from scratch, you can download the pretrained checkpoints of HAT from GoogleDrive, [Flickr30K](https://drive.google.com/file/d/11Zax1FTNnq0rcer8PxZFlx1mf7d-S67n/view?usp=drive_link)  (for Flickr30K dataset) and [MSCOCO](https://drive.google.com/file/d/1lQDeGvipaREZcwd7-owfgPidft6f4lHo/view?usp=drive_link)  (for MSCOCO dataset).

## Training
Run `run.py`:

For DAIM on Flickr30K:

```bash
python run.py with data_root=`$DATA_PATH`
```

For DAIM on MSCOCO:

```bash
python run.py with coco_config data_root=`$DATA_PATH`
```


## Testing with checkpoints

Test on Flickr30K:

```bash
python run.py with data_root=`$DATA_PATH` test_only=True checkpoint=`$CHECKPOINT_PATH`
```

Test on MSCOCO:

```bash
python run.py with coco_config data_root=`$DATA_PATH` test_only=True checkpoint=`$CHECKPOINT_PATH`
```


