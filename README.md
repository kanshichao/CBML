# Deep Bayesian Metric Learning (DBML)

Code for the TPAMI submitted paper [Deep Bayesian Metric Learning with Similarity Distribution Constraints](.)

### Performance compared with SOTA methods on CUB-200-2011

|Rank@K | 1 | 2 | 4 | 8 |
 |:---  |:-:|:-:|:-:|:-:|
|Ranked List (H) | 57.4 | 69.7 | 79.2 | 86.9 |
|Ranked List (L,M,H) | 61.3 | 72.7 | 82.7 | 89.4 |
|SoftTriple | 65.4 | 76.4 | 84.5 | 90.4 |
|DeML | 65.4 | 75.3 | 83.7 | 89.5 |
|MS | 65.7| 77.0| 86.3| 91.2|
|Contrastive+HORDE |66.8 |77.4 |85.1 |91.0 |
|**Our DBML-BN-Inception**| **69.5** |**79.4** |**87.0** |**92.4** |
|Devide-Conquer| 65.9| 76.6| 84.4| 90.6|
|MIC+Margin| 66.1| 76.8| 85.6| -|
|TML| 62.5| 73.9| 83.0| 89.4|
|**Our DBML-ResNet50**|**69.9** |**80.4** |**87.2**|**92.5** |


### Prepare the data and the pretrained model 

The following script will prepare the [CUB](http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) dataset for training by downloading to the ./resource/datasets/ folder; which will then build the data list (train.txt test.txt):

```bash
./scripts/prepare_cub.sh
```

Download the imagenet pretrained model of 
[googlenet](https://download.pytorch.org/models/googlenet-1378be20.pth), [bninception](http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth) and [resnet50](https://download.pytorch.org/models/resnet50-19c8e357.pth), and put them in the folder:  ~/.cache/torch/checkpoints/.


### Installation

```bash
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop build
```
###  Train and Test on CUB-200-2011 with DBML-Loss based on BN-Inception backbone

```bash
./scripts/run_cub_bninception.sh
```
Trained models will be saved in the ./output-bninception-cub/ folder if using the default config.

Best recall@1 higher than 69 (69.5 in the paper).

###  Train and Test on CUB-200-2011 with DBML-Loss based on ResNet50 backbone

```bash
./scripts/run_cub_resnet50.sh
```
Trained models will be saved in the ./output-resnet50-cub/ folder if using the default config.

Best recall@1 higher than 69.5 (69.9 in the paper).

### Citation

If you use this method or this code in your research, please cite as:

    @inproceedings{TPAMI-Shichao-2020,
    title={Deep Bayesian Metric Learning with Similarity Distribution Constraints},
    author={Shichao Kan, Yigang Cen, Yang Li, Zhihai He},
    booktitle={},
    pages={},
    year={2020}
    }

### Acknowledgments
This code is written based on the framework of [MS-Loss](https://github.com/MalongTech/research-ms-loss), we are really grateful to the authors of the MS paper to release their code for academic research / non-commercial use.

### License
This code is released for academic research / non-commercial use only. If you wish to use for commercial purposes, please contact [Shichao Kan](https://kanshichao.github.io) by email kanshichao10281078@126.com.

### Recomemended Papers About Metric Learning
* Michael Opitz, Georg Waltner, Horst Possegger, Horst Bischof: [Deep Metric Learning with BIER: Boosting Independent Embeddings Robustly.](https://arxiv.org/abs/1801.04815) IEEE Trans. Pattern Anal. Mach. Intell. 42(2): 276-290 (2020)
* Junnan Li, Pan Zhou, Caiming Xiong, Richard Socher, Steven C. H. Hoi: [Prototypical Contrastive Learning of Unsupervised Representations.](https://arxiv.org/abs/2005.04966) arXiv abs/2005.04966 (2020)
