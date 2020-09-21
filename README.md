# Contrastive Bayesian Analysis for Supervised Deep Metric Learning

This code is mainly for reproducing the results reported in our TPAMI submitted paper [Contrastive Bayesian Analysis for Supervised Deep Metric Learning](https://github.com/kanshichao/dbml). Beyound for this purpose, we will continue to maintain this project and provide tools for both supervised and transfer unsupervised metric learning research. Aiming to integrate various loss functions and backbones to facilitate academic research progress on deep metric learning. **Now, this project contains GoogleNet, BN-Inception, ResNet18, ResNet34, ResNet50, ResNet101 and ResNet152 backbones, and [dbml_loss with log, square root and constant](https://github.com/kanshichao/dbml/blob/master/dbml_benchmark/losses/dbml.py), [crossentropy_loss](https://en.wikipedia.org/wiki/Cross_entropy), [ms_loss](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf), [rank_loss](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper.pdf), [softtriple_loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf), [margin_loss](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Sampling_Matters_in_ICCV_2017_paper.pdf), [adv_loss](https://arxiv.org/abs/1801.04815), [proxynca_loss](https://github.com/dichotomies/proxy-nca), [npair_loss](http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective), [angular_loss](https://arxiv.org/pdf/1708.01682.pdf), [contrastive_loss](http://papers.nips.cc/paper/5416-deep-learning-face-representation-by-joint-identification-verification), [triplet_loss](https://arxiv.org/pdf/1703.07737.pdf), [cluster_loss](https://arxiv.org/pdf/1812.10325.pdf), [histogram_loss](https://arxiv.org/pdf/1611.00822.pdf), [center_loss](https://ydwen.github.io/papers/WenECCV16.pdf) and multiple losses.**

<img src="img/distribution.png" width="100%" height="65%"> 

<img src="img/constrain.png" width="100%" height="65%"> 

### Abstract
Recent methods for deep metric learning has been focusing on designing different contrastive loss functions betweenpositive and negative pairs of samples so that the learned feature embedding is able to pull positive samples of the same class closerand push negative samples from different classes away from each other. In this work, we recognize that there is a significant semanticgap between metric analysis at intermediate feature layers and class label decision at the final output layer. To bridge this gap, wedevelop a contrastive Bayesian analysis to characterize and model the posterior probabilities of image labels conditioned by their metricsimilarity in a contrastive learning setting. This contrastive Bayesian analysis leads to a new loss function for deep metric learning. Toimprove the generalization capability of the proposed method onto new classes, we further extend the contrastive Bayesian loss with amarginal variance constraint. Our experimental results and ablation studies demonstrate that the proposed contrastive Bayesian metriclearning method significantly improves the performance of deep metric learning, outperforming existing methods by a large margin.

### Performance compared with SOTA methods on CUB-200-2011 for 512 dimensional embeddings
* Googlenet Backbone

|Recall@K | 1 | 2 | 4 | 8 |
 |:---  |:-:|:-:|:-:|:-:|
|Contrastive | 26.4 | 37.7 | 49.8 | 62.3 |
|Triplet | 36.1 | 48.6 | 59.3 | 70.0 |
|LiftedStruct | 47.2 | 58.9 | 70.2 | 80.2 |
|Binomial Deviance | 52.8 | 64.4 | 74.7 | 83.9 |
|Histogram Loss| 50.3| 61.9| 72.6| 82.3|
|N-Pair-Loss | 51.0| 63.3| 74.3| 83.2|
|Clustering |48.2 |61.4 |71.8 |81.9 |
|Proxy NCA| 49.2| 61.9| 67.9| 72.4|
|Smart Mining| 49.8| 62.3| 74.1| 83.3|
|HDC| 53.6| 65.7| 77.0| 85.6|
|Angular Loss| 54.7| 66.3| 76.0| 83.9|
|BIER| 55.3| 67.2| 76.9| 85.1|
|A-BIER| 57.5| 68.7| 78.3| 82.6|
|**Ours DBML-const-GoogleNet**| **62.8** |**73.9** |**83.2** |**89.8** |
|**Ours DBML-sqrt-GoogleNet**| **63.1** |**74.7** |**83.1** |**89.8** |
|**Ours DBML-log-GoogleNet**| **63.8** |**74.8** |**83.6** |**90.3** |

* BN-Inception Backbone

|Recall@K | 1 | 2 | 4 | 8 |
 |:---  |:-:|:-:|:-:|:-:|
|Ranked List (H) | 57.4 | 69.7 | 79.2 | 86.9 |
|Ranked List (L,M,H) | 61.3 | 72.7 | 82.7 | 89.4 |
|SoftTriple | 65.4 | 76.4 | 84.5 | 90.4 |
|DeML | 65.4 | 75.3 | 83.7 | 89.5 |
|MS | 65.7| 77.0| 86.3| 91.2|
|Contrastive+HORDE |66.8 |77.4 |85.1 |91.0 |
|**Ours DBML-const-BN-Inception**| **68.3** |**78.5** |**86.9** |**92.1** |
|**Ours DBML-sqrt-BN-Inception**| **69.5** |**79.5** |**86.7** |**91.8** |
|**Ours DBML-log-BN-Inception**| **69.5** |**79.4** |**87.0** |**92.4** |

* ResNet50 Backbone

|Recall@K | 1 | 2 | 4 | 8 |
 |:---  |:-:|:-:|:-:|:-:|
|Devide-Conquer| 65.9| 76.6| 84.4| 90.6|
|MIC+Margin| 66.1| 76.8| 85.6| -|
|TML| 62.5| 73.9| 83.0| 89.4|
|**Ours DBML-const-ResNet50**|**69.2** |**79.3** |**86.3**|**91.6** |
|**Ours DBML-sqrt-ResNet50**|**70.0** |**79.9** |**87.0**|**92.0** |
|**Ours DBML-log-ResNet50**|**69.9** |**80.4** |**87.2**|**92.5** |

### Performance compared with SOTA methods on CUB-200-2011 for 64 dimensional embeddings
* ResNet50 Backbone

|Recall@K | 1 | 2 | 4 | 8 |
 |:---  |:-:|:-:|:-:|:-:|
 |N-Pair| 53.2 | 65.3| 76.0 | 84.8|
 |ProxyNCA| 55.5 | 67.7 | 78.2 | 86.2 |
 |EPSHN| 57.3| 68.9| 79.3| 87.2|
 |MS| 57.4| 69.8| 80.0| 87.8|
 |***Ours DBML-const-ResNet50***|**65.0**|**76.2**|**84.9**|**90.6**|
 |***Ours DBML-sqrt-ResNet50***|**65.0**|**76.0**|**84.1**|**90.3**|
 |***Ours DBML-log-ResNet50***|**64.3**|**75.7**|**84.1**|**90.1**|
 
 * ResNet18 Backbone
 
 |Recall@K | 1 | 2 | 4 | 8 |
 |:---  |:-:|:-:|:-:|:-:|
 |N-Pair|52.4|65.7|76.8|84.6|
 |ProxyNCA|51.5|63.8|74.6|84.0|
 |EPSHN|54.2|66.6|77.4|86.0|
 |***Ours DBML-const-ResNet18***|**58.0**|**69.6**|**80.0**|**87.5**|
 |***Ours DBML-sqrt-ResNet18***|**59.4**|**70.5**|**80.4**|**88.0**|
 |***Ours DBML-log-ResNet18***|**61.3**|**72.6**|**81.9**|**88.7**|
 
 * GoogleNet Backbone
 
 |Recall@K | 1 | 2 | 4 | 8 |
 |:---  |:-:|:-:|:-:|:-:|
 |Triplet|42.6|55.0|66.4|77.2|
 |N-Pair|45.4|58.4|69.5|79.5|
 |ProxyNCA|49.2|61.9|67.9|72.4|
 |EPSHN|51.7|64.1|75.3|83.9|
 |***Ours DBML-const-GoogleNet***|**56.8**|**69.5**|**79.5**|**87.9**|
 |***Ours DBML-sqrt-GoogleNet***|**57.7**|**69.7**|**80.5**|**88.3**|
 |***Ours DBML-log-GoogleNet***|**59.3**|**70.7**|**80.6**|**88.1**|

### Prepare the data and the pretrained model 

The following script will prepare the [CUB](http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) dataset for training by downloading to the ./resource/datasets/ folder; which will then build the data list (train.txt test.txt):

```bash
./scripts/prepare_cub.sh
```

To reproduce the results of our paper. Download the imagenet pretrained model of 
[googlenet](https://download.pytorch.org/models/googlenet-1378be20.pth), [bninception](http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth) and [resnet50](https://download.pytorch.org/models/resnet50-19c8e357.pth), and put them in the folder:  ~/.cache/torch/checkpoints/.


### Installation

```bash
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop build
```
###  Train and Test on CUB-200-2011 with DBML-Loss based on the BN-Inception backbone

```bash
./scripts/run_cub_bninception.sh
```
Trained models will be saved in the ./output-bninception-cub/ folder if using the default config.

###  Train and Test on CUB-200-2011 with DBML-Loss based on the ResNet50 backbone

```bash
./scripts/run_cub_resnet50.sh
```
Trained models will be saved in the ./output-resnet50-cub/ folder if using the default config.

###  Train and Test on CUB-200-2011 with DBML-Loss based on the GoogleNet backbone

```bash
./scripts/run_cub_googlenet.sh
```
Trained models will be saved in the ./output-googlenet-cub/ folder if using the default config.

### Citation

If you use this method or this code in your research, please cite as:

    @inproceedings{Shichao-2020,
    title={Contrastive Bayesian Analysis for Supervised Deep Metric Learning},
    author={Shichao Kan, Yigang Cen, Yang Li, Mladenovic Vladimir, and Zhihai He},
    booktitle={},
    pages={},
    year={2020}
    }

### Acknowledgments
This code is written based on the framework of [MS-Loss](https://github.com/MalongTech/research-ms-loss), we are really grateful to the authors of the MS paper to release their code for academic research / non-commercial use. We also thank the following helpful implementtaions on [histogram](https://github.com/valerystrizh/pytorch-histogram-loss), [proxynca](https://github.com/dichotomies/proxy-nca), [n-pair and angular](https://github.com/leeesangwon/PyTorch-Image-Retrieval), [siamese-triplet](https://github.com/adambielski/siamese-triplet), [clustering](https://github.com/shaoniangu/ClusterLoss-Pytorch-ReID). 

### License
This code is released for academic research / non-commercial use only. If you wish to use for commercial purposes, please contact [Shichao Kan](https://kanshichao.github.io) by email kanshichao10281078@126.com.

### Recommended Papers About Deep Metric Learning
* Michael Opitz, Georg Waltner, Horst Possegger, Horst Bischof: [Deep Metric Learning with BIER: Boosting Independent Embeddings Robustly.](https://arxiv.org/abs/1801.04815) IEEE Trans. Pattern Anal. Mach. Intell. 42(2): 276-290 (2020)
* Junnan Li, Pan Zhou, Caiming Xiong, Richard Socher, Steven C. H. Hoi: [Prototypical Contrastive Learning of Unsupervised Representations.](https://arxiv.org/abs/2005.04966) arXiv abs/2005.04966 (2020)
