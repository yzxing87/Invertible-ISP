# Invertible Image Signal Processing

**This repository will include codes for "Invertible Image Signal Processing (CVPR2021)"**

![Python 3.7](https://img.shields.io/badge/Python-3.7-green.svg?style=plastic)
![pytorch 1.1.0](https://img.shields.io/badge/Pytorch-1.1.0-green.svg?style=plastic)

![](./figures/teaser.png)
**Figure:** *Our framework*

Unprocessed RAW data is a highly valuable image format for image editing and computer vision. However, since the file size of RAW data is huge, most users can only get access to processed and compressed sRGB images. To bridge this gap, we design an Invertible Image Signal Processing (InvISP) pipeline, which not only enables rendering visually appealing sRGB images but also allows recovering nearly perfect RAW data. Due to our framework's inherent reversibility, we can reconstruct realistic RAW data instead of synthesizing RAW data from sRGB images, without any memory overhead. We also integrate a differentiable JPEG compression simulator that empowers our framework to reconstruct RAW data from JPEG images. Extensive quantitative and qualitative experiments on two DSLR demonstrate that our method obtains much higher quality in both rendered sRGB images and reconstructed RAW data than alternative methods. 

> **Invertible Image Signal Processing** <br>
>  Yazhou Xing*, Zian Qian*, Qifeng Chen <br>
>  HKUST <br>

[[Paper](https://github.com/yzxing87/Invertible-ISP)] 
[[Project Page](https://yzxing87.github.io/InvISP/index.html)]
[[Technical Video (Coming soon)](https://yzxing87.github.io/TBA)]

![](./figures/result_01.png)
**Figure:** *Our results*


## Code will come soon. 


## Citation

```
@inproceedings{xing21invertible,
  title     = {Invertible Image Signal Processing},
  author    = {Xing, Yazhou and Qian, Zian and Chen, Qifeng},
  booktitle = {CVPR},
  year      = {2021}
}
```
