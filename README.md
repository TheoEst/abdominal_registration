# Symmetric deep learning based registration using noisy segmentation labels (Learn2Reg Task 3: CT Abdominal)


This repository corresponds to the 2nd ranked for Task 3 (CT Abdominal) and 2nd overall method for the Learn2Reg Challenge 2020 : https://learn2reg.grand-challenge.org/.

You can also consult the repository for the Task 3 : https://github.com/TheoEst/hippocampus_registration.
Implementation is made with Pytorch.

## Use this repository

In order to use this repository, you only need to download the Learn2Reg Task 3 Data : https://learn2reg.grand-challenge.org/Datasets/ and add it on the ./data/ folder. 


## Methodology 

Our method is based on the article  *Deep Learning-Based Concurrent Brain Registration and Tumor Segmentation*, **Estienne T., Lerousseau M. et al.**, 2020 (https://www.frontiersin.org/articles/10.3389/fncom.2020.00017/full).


In this work we proposed a deep learning based registration using 3D Unet as backbone with 3 losses :
* Reconstruction loss ( Mean Square Error or Local Cross Correlation)
* Segmentation loss ( Dice Loss between deformed segmentation and ground truth segmentation)
* Regularisation loss (To force smoothness)

In the proposed architecture, the moving and fixed image are passed independently through the encoder, and then merged with subtraction operation.

<p align="center">
<img src="https://github.com/TheoEst/hippocampus_registration/blob/main/method.PNG" width="750">
</p>
  
## Models

5 pretrained models are available on the ./models folder : 
* Baseline model
* Baseline model with symmetric training 
* Baseline model with pretraining model (unsupervised pretraining)
* Baseline model with pretraining model (supervised pretraining with noisy segmentations)
* Baseline model with pretraining model (supervised pretraining with noisy segmentations), trained with both training and validation dataset (used for the test submission)


To recreate this models, launch the following commands :

``` 

```

## Prediction

To predict, use the *predict_reg.py* file. 

```
Options : 
  --val                 Do the inference for the validation dataset
  --train               Do the inference for the train dataset
  --test                Replace the validation dataset by test set. (--val is necessary)
  --save-submission     Save the submission in the format for the Learn2Reg challenge
  --save-deformed-img   Save the deformed image and deformed mask in numpy format
  --save-grid           Save the grid in numpy format

Examples :

```

## Create submission & evaluation 


## Performances 

Results on the validation set 

  
Method | Dice | Dice 30 | Hausdorff Distance
------------ | ------------- | ------------ | ------------- 
Baseline  | 0.396 | 0.348 | 16.6 
Baseline + sym.  | 0.411 | 0.367 | 17.1
Baseline + sym. + pretrain | 0.538 | 0.511  | 13.6 
Baseline + sym. + pretrain + noisy labels | **0.633** |**0.613**  | **10.4** 


Example of the results on the validation set :

<p align="center">
<img src="https://github.com/TheoEst/hippocampus_registration/blob/main/results.png" width="500">
</p>


## Developer

This package was developed by Théo Estienne<sup>12</sup>


<sup>1</sup> Université Paris-Saclay, **CentraleSupélec**, *Mathématiques et Informatique pour la Complexité et les Systèmes*, 91190, Gif-sur-Yvette, France.

<sup>2</sup> Université Paris-Saclay, **Institut Gustave Roussy**, Inserm, *Radiothérapie Moléculaire et Innovation Thérapeutique*, 94800, Villejuif, France.
