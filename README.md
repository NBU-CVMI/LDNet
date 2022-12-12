# LDNet: Lightweight dynamic convolution network for human pose estimation

## Introduction
This is an official pytorch implementation of [LDNet: Lightweight dynamic convolution network for human pose estimation ](https://www.sciencedirect.com/science/article/pii/S1474034622002439).We present a lighter model compared with existing pose estimation models, called LDNet. In order to construct it, we start from a lightweight feature extraction head to reduce the number of image preprocessing parameters. Then, we employ a high-resolution parallel sub-network to predict precise keypoint heatmaps. Moreover, we propose a lightweight dynamic convolution. It can cope with changing human poses by adaptively learning different convolution parameters. Finally, to further exploit the relationship between the high-level semantic and spatial structure features for accurately locating different keypoints, we propose a keypoint refinement module based on our lightweight dynamic convolution to improve the keypoint detection and location results. 



## Network Structure

![Illustrating the architecture of LDNet](https://github.com/NBU-CVMI/LDNet/blob/main/imgs/the_structure_of_LDNet.png)



## Compartive result with other models
### Results on COCO val  set
![image](imgs/result_on_coco_val.png)

### Results on COCO test-dev set
![image](imgs/result_on_coco_test_dev.png)


### Results  on MPII val set
![image](imgs/result_on_mpii_val.png)

### Results  on CrowdPose test set

![image](imgs/result_on_crowdpose.png)



## Environment

We use  Pytorch on ubuntu18.04 and 1 NVIDIA GeForce 2080Ti to train our models.



### Citation
If you use our code or models in your research, please cite with:
```
@article{xu2022ldnet,
  title={LDNet: Lightweight dynamic convolution network for human pose estimation},
  author={Xu, Dingning and Zhang, Rong and Guo, Lijun and Feng, Cun and Gao, Shangce},
  journal={Advanced Engineering Informatics},
  volume={54},
  pages={101785},
  year={2022},
  publisher={Elsevier}
}
```