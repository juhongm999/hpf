## Hyperpixel Flow: <br/> Semantic Correspondence with Multi-layer Neural Features
This is the implementation of the paper "Hyperpixel Flow: Semantic Correspondence with Multi-layer Neural Features" by J. Min, J. Lee, J. Ponce and M. Cho.
Implemented on Python 3.6 and Pytorch 1.0.1.

![](https://juhongm999.github.io/pic/hpf.png)

For more information, check out project [[website](http://cvlab.postech.ac.kr/research/HPF/)] and the paper on [[arXiv](http://arxiv.org/abs/1908.06537)].

### Conda environment settings

    conda create -n hpf python=3.6
    conda activate hpf

    cat /usr/local/cuda/version.txt
    conda install pytorch=1.0.1 torchvision cudatoolkit=10.0 -c pytorch (if CUDA 10) 
    conda install pytorch=1.0.1 torchvision cudatoolkit=9.0 -c pytorch (if CUDA 9) 
    
    conda install -c anaconda scikit-image
    conda install -c anaconda pandas
    conda install -c anaconda requests
    conda install pillow=6.1
    pip install gluoncv-torch

### Reproduction    

Beam search on SPair-71k validation set: 

    python beamsearch.py --dataset spair --thres bbox --backbone resnet50
    python beamsearch.py --dataset spair --thres bbox --backbone resnet101
    
    
Beam search on PF-PASCAL validation set: 

    python beamsearch.py --dataset pfpascal --thres bbox --backbone resnet50
    python beamsearch.py --dataset pfpascal --thres bbox --backbone resnet101  
    
    
Results on PF-PASCAL: (PCK: 83.4%, 84.8%, 88.3%)

    python evaluate.py --dataset pfpascal --backbone resnet50 --hyperpixel '(2,7,11,12,13)'
    python evaluate.py --dataset pfpascal --backbone resnet101 --hyperpixel '(2,17,21,22,25,26,28)'
    python evaluate.py --dataset pfpascal --backbone fcn101 --hyperpixel '(2,4,5,18,19,20,24,32)'

Results on PF-WILLOW: (PCK: 74.4%)

    python evaluate.py --dataset pfwillow --backbone resnet101 --hyperpixel '(2,17,21,22,25,26,28)'

Results on Caltech-101: (LT-ACC: 0.88, IoU: 0.64)

    python evaluate.py --dataset caltech --backbone resnet50 --hyperpixel '(2,7,11,12,13)'

Results on SPair-71k: (PCK: 27.2%, 28.2%)
 
    python evaluate.py --dataset spair --backbone resnet50 --hyperpixel '(0,9,10,11,12,13)'
    python evaluate.py --dataset spair --backbone resnet101 --hyperpixel '(0,8,20,21,26,28,29,30)'
    
To visualize predictions using TPS transformation, add command line argument **--visualize**: 

    python evaluate.py --visualize
    
### BibTeX
If you use this code and SPair-71k dataset for your research, please consider citing:
````BibTeX
@InProceedings{min2019hyperpixel, 
   title={Hyperpixel Flow: Semantic Correspondence with Multi-layer Neural Features},
   author={Juhong Min and Jongmin Lee and Jean Ponce and Minsu Cho},
   booktitle={ICCV},
   year={2019}
}
````
````BibTeX
@article{min2019spair,
   title={SPair-71k: A Large-scale Benchmark for Semantic Correspondence},
   author={Juhong Min and Jongmin Lee and Jean Ponce and Minsu Cho},
   journal={arXiv prepreint arXiv:1908.10543},
   year={2019}
}
````
