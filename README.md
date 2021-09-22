# Contrastive-Finetuning
This repo is the official implementation of the following paper:
**"On the Importance of Distractors for Few-Shot Classification"** [Paper](https://arxiv.org/abs/2109.09883)

If you find this repo useful for your research, please consider citing this paper  
```
@misc{das2021importance,
    title={On the Importance of Distractors for Few-Shot Classification},
    author={Rajshekhar Das and Yu-Xiong Wang and JoséM. F. Moura},
    year={2021},
    eprint={2109.09883},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
# Dataset Download

To set up the dataset, follow the exact steps outlined in [here](https://github.com/hytseng0509/CrossDomainFewShot#datasets).

# Pretrained Model

To download the pretrained backbone model, follow the exact steps outlined in [here](https://github.com/hytseng0509/CrossDomainFewShot#feature-encoder-pre-training)

# Running

* To run contrastive finetuning on `cub` data (default target domain) with the downloaded pretrained model, simply run ```bash conft.sh```  
* To run the multi-task variant on the same target domain, run  ```bash mt_conft.sh```  
* To change the target domain or other hyperparameters, refer to `conft.sh` and `mt_conft.sh`

# Acknowlegements

Part of the codebase, namely, the dataloaders have been adapted from [Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation](https://github.com/hytseng0509/CrossDomainFewShot#datasets).

