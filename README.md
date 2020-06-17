# Hippocampus Segmentation from MRI using V-Net
In this repo, hippocampus segmentation from MRI is performed 
using a Convolutional Neural Network (CNN) architecture based on
[V-Net](https://arxiv.org/abs/1606.04797).
The dataset is publicly available from the 
[Medical Segmentation Decathlon Challenge](http://medicaldecathlon.com/),
and can be downloaded from 
[here](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2).
The [PyTorch](https://pytorch.org/) library has been used to write the model architecture 
and performing the training and validation. [SimpleITK](https://simpleitk.org/) 
has been exploited to handle I/O of medical images.
A 5-folders cross validation has been performed on the training set, yielding a 
Mean Multi Dice Coefficient of 0.8789 +/- 0.0211.
The result is reported as "mean +/- std". This result has been calculated by 
considering batches of 4 images in the cross-validation process.
Meshes and images reported in the ```images``` folder have been obtained exploiting 
[ITK-SNAP](http://www.itksnap.org/).

### TODO List
- [x] CNN Architecture Definition
- [x] 3D Data Loader for Nifti files
- [x] Definition of loss functions
- [x] Training loop
- [x] Cross-validation
- [ ] Data Augmentation
- [ ] Test
- [ ] Command Line Interface for training and testing 

## Training
If you simply want to perform the training, run the train.py file.
If you want to edit the configuration, modify the config.py file.
In particular, consider the class ```SemSegMRIConfig```.

## Testing
If you want to perform the inference, either on the training set images, or the
test set images, see the test.py file.

### Sample Images (Training set)
#### Ground Truth Images 
![Ground Truth - MRI 327 (1)]("./images/327_gt_01.png")
![Ground Truth - MRI 327 (2)](“./images/327_gt_02.png”)
#### Predicted Images 
![Prediction   - MRI 327 (1)]("./images/327_pred_01.png")
![Prediction   - MRI 327 (2)](“./images/327_pred_02.png”)
 
### Sample Images (Test set)
#### Predicted Images
![Prediction   - MRI 283 (1)]("./images/283_pred_01.png")
![Prediction   - MRI 283 (2)](“./images/283_pred_02.png")