# Hierarchical U-Shape Attention Network for Salient Object Detection

by Sanping Zhou, Jinjun Wang, Jimuyang Zhang, Le Wang, Dong Huang, and Nanning Zheng. [[paper link](https://ieeexplore.ieee.org/document/9152130)]

This implementation is written by Sanping Zhou at Xi'an Jiaotong University.

***

## Citation
@article{zhou2020hierarchical,
&nbsp;&nbsp;&nbsp;&nbsp;  title={Hierarchical U-Shape Attention Network for Salient Object Detection},
&nbsp;&nbsp;&nbsp;&nbsp;  author={Zhou, Sanping and Wang, Jinjun and Zhang, Jimuyang and Wang, Le and Huang, Dong and Du, Shaoyi and Zheng, Nanning},
&nbsp;&nbsp;&nbsp;&nbsp;  journal={IEEE Transactions on Image Processing},
&nbsp;&nbsp;&nbsp;&nbsp;  volume={29},
&nbsp;&nbsp;&nbsp;&nbsp;  pages={8417--8428},
&nbsp;&nbsp;&nbsp;&nbsp;  year={2020},
&nbsp;&nbsp;&nbsp;&nbsp;  publisher={IEEE}
}

## Requirement
* Python 2.7
* PyTorch 0.4.0
* torchvision
* numpy
* Cython
* pydensecrf ([here](https://github.com/Andrew-Qibin/dss_crf) to install)

## Training
1. Set the path of pretrained ResNeXt model in resnext/config.py
2. Set the path of DUTS dataset in config.py
3. Run by ```python train_HUAN3.py```

*Hyper-parameters* of training were gathered at the beginning of *train_HUAN3.py* and you can conveniently 
change them as you need.

## Testing
1. Set the path of five benchmark datasets in config.py
2. Put the trained model in ckpt/R3Net
2. Run by ```python infer.py```

*Settings* of testing were gathered at the beginning of *infer.py* and you can conveniently 
change them as you need.

## Useful links
* [MSRA10K](http://mmcheng.net/msra10k/): our training set
* [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), 
[HKU-IS](https://sites.google.com/site/ligb86/hkuis), 
[PASCAL-S](http://cbi.gatech.edu/salobj/), 
[SOD](http://elderlab.yorku.ca/SOD/), 
[DUT-OMRON](http://ice.dlut.edu.cn/lu/DUT-OMRON/Homepage.htm): the five benchmark datasets
