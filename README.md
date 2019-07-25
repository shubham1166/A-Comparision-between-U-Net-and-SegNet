# A comparison between SegNet and U-Net
---

In this repository, i have tried to impliment SegNet and U-Net and  tried to compare their results that we have got for segmentation.

---
## Dataset
The dataset that I have used is a kaggle dataset for identification and segmentation of nuclei in cells. The dataset consists of 670 images and each of the image is an RGB image with dimension 512×512. Below is the figue of the type of data that we have for segmentation.
**![](https://lh6.googleusercontent.com/Ngzs_qC2dUCs-fRkOOVSumBDYS8R3KI69cVdTWaQA6SxM2Qmlsh6tr39SlN5R_6kn_iV_l3xiAS6B6Lwvl96LL_Yzwj18t3c1H0JSyzHDlt4Q7aRoD2I1qkzjgeXUDnq_HcpO5wR)**
**Fig**: Row1 consists of original images followed by row2 that consists of original labelled images(Source:[Kaggle](https://www.kaggle.com/paultimothymooney/identification-and-segmentation-of-nuclei-in-cells))

---
## Segmentation by SegNet
The network architecture that i have used here has VGG-166 in downsampling. The network architecture consists of two convolutional layers of kernel (3,3) followed by a maxpool. 
**![](https://lh3.googleusercontent.com/-owN_ZZaV_PpPvqsyGnYLChOyMP-r3TJlk7U5pQibBvtGj4FzjLlXlUKO15RNV1VGUSzUo8UV8LeM1k7lJvgXPQ-qfp_SQ-LoTnzmAGBC92xrdKXIbwHvrTmj7N2G1jLV5DFre-n)**
**Fig**: SegNet architecture; Source: google

---
## Segmentation by U-Net
The main difference of SegNet and U-net is that we use concatenation of layers in case of U-Net that helps the network to learn better while upsampling the data. Below is the figure of a simple U-Net architecture:
![Image result for U-Net architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)
**Fig**: An architecture of a U-Net Network; Source:google
---
## Results
![](https://www.overleaf.com/project/5d1dcc6dbb53f75109902303/file/5d2c13b24a4c0d57d252f3f7)
**Fig**: Row1: Input data,Row2: SegNet output,Row3: U-Net output,Row4: Original labels

**![](https://lh5.googleusercontent.com/4rRQVClcGvy0PuSb05USTuJ6pjCkXc1GwfOqDDxEgcWaepppmSmWJEriuKJp4iCDx1W2Hu84eYAf7t3PwraxCCcWi0LYNLG3M56kycVKyWqEhimPRByle-mrigZkEz2t7gro3aoa)**
