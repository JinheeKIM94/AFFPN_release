# Attentive Feedback Feature Pyramid for Shadow Detection 

This repository is a Pytorch implementation of the paper [**"Attentive Feedback Feature Pyramid for Shadow Detection"**](https://ieeexplore.ieee.org/document/9242301)

Jinhee Kim and [Wonjun Kim](https://sites.google.com/view/dcvl)  
IEEE Signal Processing Letters

When using this code in your research, please cite the following paper: 

Jinhee Kim and Wonjun Kim, **"Attentive Feedback Feature Pyramid for Shadow Detection,"** **IEEE Signal Processing Letters** vol. 27, pp. 1964-1968, Oct. 2020.

```  
@ARTICLE{9242301,
author={J. {Kim} and W. {Kim}},
journal={IEEE Signal Processing Letters}, 
title={Attentive Feedback Feature Pyramid Network for Shadow Detection}, 
year={2020},
volume={27},
number={},
pages={1964-1968},}
```  
### Model architecture
![network](https://user-images.githubusercontent.com/60129726/97146672-944f3f00-17ab-11eb-9421-413bb2840c00.png)

### Experimental results with state-of-the-arts methods

![fig2_git](https://user-images.githubusercontent.com/60129726/80967835-f90b8b80-8e51-11ea-9b60-11e72f50a6cd.png)
Several results of shadow detection based on ISTD dataset. 1st column: shadow images. 2nd-8th columns: detection results by Hu *et al.*, Chen *et al.*, Wang *et al.*, Hu *et al.*, Zhu *et al.*, Zheng *et al.*, and the proposed method. 9th column: ground truth

![fig2](https://user-images.githubusercontent.com/60129726/80562585-1213d700-8a23-11ea-86e5-a75519bc322e.png)
Several results of shadow detection based on SBU and UCF datasets. 1st column: shadow images. 2nd-8th columns: detection results by Hu *et al.*, Chen *et al.*, Le *et al.*, Hu *et al.*, Zhu *et al.*, Zheng *et al.*, and the proposed method. 9th column: ground truth

### Requirements

* Python >= 3.5
* Pytorch 1.3.1
* Ubuntu 16.04
* CUDA 9.2 (if CUDA available)
* cuDNN (if CUDA available)

### Pretrained models
You can download pretrained ResNext101 weight
* [Download ResNext101](https://drive.google.com/drive/folders/1bTx9eB5_tcMYHSVAFyTpTHUDS_PHEIvg?usp=sharing)

### Pretrained models
You can download pretrained AFFPN model
* [Trained with AFFPN](https://drive.google.com/drive/folders/1cm4CmxCBoqVJlom5WCuL-mvLyf4-Jd_m?usp=sharing)

### Test result
You can download test results of our AFFPN Model
* [Test resut with AFFPN](https://drive.google.com/drive/folders/1yRrbVLmDZPY6VBG7IfSoXgbiQqrEwNVI?usp=sharing)

### Note 
1. You should place the weights of ResNext101 in the ./resnext/
2. You should place the weights of AFFPN in the ./trained/[dataset_name]/  
3. Dataset is also should placed in the ./dataset directory  
4. Test results are saved in the ./output/[dataset_name]/
5. You can adjust the detailed settings in config.py

### Training
* Attentive Feedback Feature Pyramid for Shadow Detection network training
```bash
python train_AFFPN.py
```
## Testing 
* Attentive Feedback Feature Pyramid for Shadow Detection  network testing
```bash
python test_AFFPN.py
```
