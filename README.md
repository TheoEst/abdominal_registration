# Symmetric deep learning based registration using noisy segmentation labels (Learn2Reg Task 3: CT Abdominal)


This repository corresponds to the 2nd ranked for Task 3 (CT Abdominal) and 2nd overall method for the Learn2Reg Challenge 2020 : https://learn2reg.grand-challenge.org/.

You can also consult the repository for the Task 3 : https://github.com/TheoEst/hippocampus_registration.
Implementation is made with Pytorch.

## Data

In order to use this repository, you only need to download the Learn2Reg Task 3 Data : https://learn2reg.grand-challenge.org/Datasets/ and add it on the ./data/ folder. 

If you want to use the supplementary data, you need to download the following datasets : 
* Medical Segementation Decathlon (http://medicaldecathlon.com/). We use 5 cohorts from the MSD dataset : Liver, Pancreas, Colon, Hepatic Vessels and Spleen.  
* KITS 19 (https://github.com/neheller/kits19).
* TCIA Pancreas (https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT and https://zenodo.org/record/1169361#.X4Vr2efgqUk). 

For each of the supplementary dataset, you need to run the two code  ```./preprocessing/resample_cohort_name.py ``` and ```./preprocessing/ants_warped_cohort_name.py ```. Theses preprocessing steps will perform resampling to 2 mm voxel and linear registration with **AntsPy** (https://github.com/ANTsX/ANTsPy).

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
(Models will be released soon). 

5 pretrained models are available on the ./models folder : 
* Baseline model
* Baseline model with symmetric training 
* Baseline model with pretraining model (unsupervised pretraining)
* Baseline model with pretraining model (supervised pretraining with noisy segmentations)
* Baseline model with pretraining model (supervised pretraining with noisy segmentations), trained with both training and validation dataset (used for the test submission)

To recreate this models, launch the following commands :

``` 
python3 -m abdominal_registration.main --crop-size 128 128 128 --val-crop-size 192 160 192 --zeros-init --workers=4 --batch-size=2 --val-batch-size 1 --epochs=300 --session-name=Baseline --lr=1e-4 --instance-norm --regu-deformable-loss-weight=1e-2 --mse-loss-weight=1 --deformed-mask-loss --random-crop --classic-vnet --channel-multiplication 8 --deep-supervision --multi-windows --cohorts learn2reg_task3 --keep-all-label

python3 -m abdominal_registration.main --crop-size 128 128 128 --val-crop-size 192 160 192 --zeros-init --workers=4 --batch-size=2 --val-batch-size 1 --epochs=300 --session-name=Baseline+symmetric --lr=1e-4 --instance-norm --regu-deformable-loss-weight=1e-2 --mse-loss-weight=1 --deformed-mask-loss --random-crop --classic-vnet --channel-multiplication 8 --deep-supervision --multi-windows --cohorts learn2reg_task3 --keep-all-label --symmetric-training

python3 -m abdominal_registration.main --crop-size 128 128 128 --val-crop-size 192 160 192 --zeros-init --workers=4 --batch-size=2 --val-batch-size 1 --epochs=300 --session-name=Baseline+symmetric+pretrain --lr=1e-4 --instance-norm --regu-deformable-loss-weight=1e-2 --mse-loss-weight=1 --deformed-mask-loss --random-crop --classic-vnet --channel-multiplication 8 --deep-supervision --multi-windows --cohorts learn2reg_task3 --keep-all-label  --symmetric-training --model-abspath ./abdominal_registration/save/models/Pretrain_unsupervised.pth.tar

python3 -m abdominal_registration.main --crop-size 128 128 128 --val-crop-size 192 160 192 --zeros-init --workers=4 --batch-size=2 --val-batch-size 1 --epochs=300 --session-name=Baseline+symmetric+pretrain_noisy_labels --lr=1e-4 --instance-norm --regu-deformable-loss-weight=1e-2 --mse-loss-weight=1 --deformed-mask-loss --random-crop --classic-vnet --channel-multiplication 8 --deep-supervision --multi-windows --cohorts learn2reg_task3 --keep-all-label --symmetric-training --model-abspath ./abdominal_registration/save/models/Pretrain_supervised.pth.tar

python3 -m abdominal_registration.main --crop-size 128 128 128 --val-crop-size 192 160 192 --zeros-init --workers=4 --batch-size=2 --val-batch-size 1 --epochs=300 --session-name=Test_submission--lr=1e-4 --instance-norm --regu-deformable-loss-weight=1e-2 --mse-loss-weight=1 --deformed-mask-loss --random-crop --classic-vnet --channel-multiplication 8 --deep-supervision --multi-windows --cohorts learn2reg_task3 --keep-all-label --symmetric-training --merge-train-val --model-abspath ./abdominal_registration/save/models/Pretrain_supervised.pth.tar
```

## Pretrain 
Two pretraining models are given : unsupervised pretraining and supervised with noisy labels.  The pretraining consist in training with the supplementary datasets and used the noisy labels (called pseudo labels in the code). 

To recreate this models, launch the following commands :

```
python3 -m abdominal_registration.main --crop-size 128 128 128 --val-crop-size 192 160 192 --zeros-init --workers=4 --batch-size=2 --val-batch-size 1 --epochs=300 --session-name=Pretrain_unsupervised --lr=1e-4 --instance-norm --regu-deformable-loss-weight=1e-2 --mse-loss-weight=1 --random-crop --classic-vnet --channel-multiplication 8 --deep-supervision --multi-windows --symmetric-training --cohorts learn2reg_task3 kits liver tcia_pancreas spleen colon pancreas hepatic --val-cohorts learn2reg_task3 --pseudo-labels


python3 -m abdominal_registration.main --crop-size 128 128 128 --val-crop-size 192 160 192 --zeros-init --workers=4 --batch-size=2 --val-batch-size 1 --epochs=300 --session-name=Pretrain_supervised --lr=1e-4 --instance-norm --regu-deformable-loss-weight=1e-2 --mse-loss-weight=1 --random-crop --classic-vnet --channel-multiplication 8 --deep-supervision --multi-windows --symmetric-training --cohorts learn2reg_task3 kits liver tcia_pancreas spleen colon pancreas hepatic --val-cohorts learn2reg_task3 --pseudo-labels --deformed-mask-loss
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

python3 -m abdominal_registration.predict_reg --crop-size 256 160 192 --batch-size=1 --val-batch-size 1  --instance-norm --classic-vnet --channel-multiplication 8 --deep-supervision --multi-windows --cohorts learn2reg_task3  --cohorts learn2reg_task3 --val --train --save-submission --model-abspath ./abdominal_registration/save/models/Baseline+symmetric+pretrain.pth.tar

```
## Segmentation

We developped a segmentation network based on a 3D Unet in order to segment 11 abdominal organs. The goal is to predict the segmentations for the supplementary data and used its for the pretraining. Our model segment the 11 following organs : Spleen, Right & Left Kidney, Liver, Stomach, Pancreas, Gall Bladder, Aorta, Inferior Vena Cava, Portal & Splenic Vein, Esophagus.
The segmentation network was trained with the following cohorts : Learn2Reg Task3, TCIA Pancreas, KITS 19 and MSD (Liver, Spleen and Pancreas). 
We used a modified dice loss function to train our network, such that we backpropagate only the loss for the organs contained in the cohort. 


Two scripts are available to train the segmentation model : *main_seg.py* and *predict_seg.py*. To recreate our experiments, launch the following commands :

```
python3 -m abdominal_registration.main_seg  --batch-size=3 --crop-size 144 144 144 --cohorts learn2reg_task3 liver pancreas spleen tcia_pancreas kits --random-crop --lr=1e-4 --instance-norm --session-name Train_seg_multi_cohorts --val-crop 192 160 192 --classic-vnet --data-augmentation --epochs=400 --val-cohorts tcia_pancreas learn2reg_task3

python3 -m abdominal_registration.predict_seg  --batch-size=1 --crop-size 256 160 192 --cohorts learn2reg_task3 liver pancreas spleen tcia_pancreas kits hepatic colon --instance-norm --classic-vnet --train --val --model-abspath ./abdominal_registration/save/models/Train_seg.pth.tar
```

## Create submission & evaluation 

To transform the predicted data into a compressed file, just use the *create_submission.py* file. For instance ```python3 ./submission/create_submission.py ./save/submission/Baseline+symmetric+pretrain ```. You will obtain a folder called  *Baseline+symmetric+pretrain_compressed* and a zip file *Baseline+symmetric+pretrain_submission* which you can submit. 

To evaluate the performance, you need just to run the *apply_evaluation.py* file. For instance ```python3 ./submission/apply_evaluation.py Baseline+symmetric+pretrain_compressed``` will generate a csv file in the *./save/evaluation/* folder with all the metrics for each pairs (Dice, Dice30, Hausdorff and standard deviation of Jacobian).

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
<img src="https://github.com/TheoEst/abdominal_registration/blob/main/results.png" width="500">
</p>


## Developer

This package was developed by Théo Estienne<sup>12</sup>


<sup>1</sup> Université Paris-Saclay, **CentraleSupélec**, *Mathématiques et Informatique pour la Complexité et les Systèmes*, 91190, Gif-sur-Yvette, France.

<sup>2</sup> Université Paris-Saclay, **Institut Gustave Roussy**, Inserm, *Radiothérapie Moléculaire et Innovation Thérapeutique*, 94800, Villejuif, France.

## References
