# EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning
EdgeConnect: Structure Guided Image Inpainting using Edge Prediction, ICCV 2019 w/ some tuning of mine.

It is based on [original repo](https://github.com/knazeri/edge-connect).

**WIP**

# Introduction
We develop a new approach for image inpainting that does a better job of reproducing filled regions exhibiting fine details 
inspired by our understanding of how artists work: lines first, color next. 
We propose a two-stage adversarial model EdgeConnect that comprises of an edge generator followed by an image completion network. 
The edge generator hallucinates edges of the missing region (both regular and irregular) of the image, 
and the image completion network fills in the missing regions using hallucinated edges as a priority. 
Detailed description of the system can be found in our [paper](https://arxiv.org/abs/1901.00212).

# Requirements
1) Python 3.x
2) Pytorch 1.x (maybe 0.x)

# Usage
0. Clone this repo
```
$ git clone https://github.com/kozistr/improved-edge-connect
$ cd improved-edge-connect
```

1. Install the dependencies
```
$ pip3 install -r requirements.txt
```

# License
Licensed under a Creative Commons Attribution-NonCommercial 4.0 International.

Except where otherwise noted, this content is published under a CC BY-NC license, 
which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes
 and give appropriate credit and provide a link to the license.

# Citation
```
@inproceedings{nazeri2019edgeconnect,
  title={EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning},
  author={Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  journal={arXiv preprint},
  year={2019},
}
```

# Author
HyeongChan Kim / [kozistr](http://kozistr.tech)
