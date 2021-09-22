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

# Reproducing Results From The Paper

To run contrastive finetuning on `cub` data (default target domain), simply run ```bash main.sh```  
For other options on target domain, finetuning mode or other hyperparameters, refer to `main.sh`

# Acknowlegements

Part of the codebase, namely, the dataloaders have been adapted from [Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation](https://github.com/hytseng0509/CrossDomainFewShot#datasets).

