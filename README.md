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
Mean Multi Dice Coefficient of 0.8719 +/- 0.0387, a Dice Coefficient for 
Anterior Hippocampus of 0.8821 +/- 0.0367 and a Dice Coefficient for 
Posterior Hippocampus of 0.8617 +/- 0.0482.
The results are reported as "mean +/- std". 
Meshes and images reported in the ```images``` folder have been obtained exploiting 
[ITK-SNAP](http://www.itksnap.org/).

### Quality Measures
<table>
<tr>
<th colspan="4">Results</th>
</tr>
<tr>
<th> Model </th>
<th> Mean Multi Dice </th>
<th> Dice (Anterior  Hippocampus) </th>
<th> Dice (Posterior Hippocampus) </th>
</tr>
<tr>
<td>3D V-Net (No Data Augmentation)</td>
<td>0.8719 +/- 0.0387</td>
<td>0.8821 +/- 0.0367</td>
<td>0.8617 +/- 0.0482</td>
</tr>
</table>

### Confusion Matrix
<table>
<tr>
<th>Confusion Matrix</th>
<th>Normalized Confusion Matrix</th>
</tr>
<tr>
<th>
<img src="images/conf_matrix_no_norm.png" alt="Confusion Matrix (Cross-validation)" width="400"/>
</th>
<th>
<img src="images/conf_matrix_normalized_row.png" alt="Normalized Confusion Matrix (Cross-validation)" width="400"/>
</th>
<tr>
</table>

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
If you simply want to perform the training, run the ```train.py``` file.
If you want to edit the configuration, modify the ```config.py``` file.
In particular, consider the class ```SemSegMRIConfig```.

## Testing
If you want to perform the inference, either on the training set images, or the
test set images, see the ```test.py``` file.

### Sample Images (Training set)
#### Ground Truth Images
<table>
<tr>
<th>Ground Truth - MRI 327 (1)</th>
<th>Ground Truth - MRI 327 (2)</th>
</tr>
<tr>
<td><img src="images/327_gt_01.png" alt="Ground Truth - MRI 327 (1)" width="250"/></td>
<td><img src="images/327_gt_02.png" alt="Ground Truth - MRI 327 (2)" width="250"/></td>
</tr>
</table>

#### Predicted Images
<table>
<tr>
<th>Prediction   - MRI 327 (1)</th>
<th>Prediction   - MRI 327 (2)</th>
</tr>
<tr>
<td><img src="images/327_pred_01.png" alt="Prediction   - MRI 327 (1)" width="250"/></td>
<td><img src="images/327_pred_02.png" alt="Prediction   - MRI 327 (2)" width="250"/></td>
</tr>
</table>

### Sample Images (Test set)
#### Predicted Images
<table>
<tr>
<th>Prediction   - MRI 283 (1)</th>
<th>Prediction   - MRI 283 (2)</th>
</tr>
<tr>
<td><img src="images/283_pred_01.png" alt="Prediction   - MRI 283 (1)" width="250"/></td>
<td><img src="images/283_pred_02.png" alt="Prediction   - MRI 283 (2)" width="250"/></td>
</tr>
</table>