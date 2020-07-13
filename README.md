# Hippocampus Segmentation from MRI using V-Net
In this repo, hippocampus segmentation from MRI is performed 
using a Convolutional Neural Network (CNN) architecture based on
[V-Net](https://arxiv.org/abs/1606.04797).
The dataset is publicly available from the 
[Medical Segmentation Decathlon Challenge](http://medicaldecathlon.com/),
and can be downloaded from 
[here](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2).

The [PyTorch](https://pytorch.org/) library has been used to write the model architecture 
and performing the training and validation. 
[SimpleITK](https://simpleitk.org/) has been exploited to handle I/O of medical images. 
3D Data Augmentation has been made by employing [torchio](https://arxiv.org/abs/2003.04696).

A 5-folders cross validation has been performed on the training set, yielding a 
Mean Multi Dice Coefficient of 0.8727 +/- 0.0364, a Dice Coefficient for 
Anterior Hippocampus of 0.8821 +/- 0.0363 and a Dice Coefficient for 
Posterior Hippocampus of 0.8634 +/- 0.0415.
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
<th> Mean Dice per case </th>
<th> Dice per case (Anterior) </th>
<th> Dice per case (Posterior) </th>
</tr>
<tr>
<td>3D V-Net (no data augmentation)</td>
<td>0.8727 +/- 0.0364</td>
<td>0.8821 +/- 0.0363</td>
<td>0.8634 +/- 0.0415</td>
</tr>
<tr>
<td>3D V-Net (with data augmentation)</td>
<td> 0.8761 +/- 0.0374 </td>
<td> 0.8875 +/- 0.0354 </td>
<td> 0.8647 +/- 0.0455 </td>
</tr>
</table>

### Confusion Matrices
#### No Data Augmentation
<table>
<tr>
<th>Confusion Matrix </th>
<th>Normalized Confusion Matrix</th>
</tr>
<tr>
<th>
<img src="images/conf_matrix_no_norm_no_augm_torchio.png" alt="Confusion Matrix (Cross-validation)" width="400"/>
</th>
<th>
<img src="images/conf_matrix_normalized_row_no_augm_torchio.png" alt="Normalized Confusion Matrix (Cross-validation)" width="400"/>
</th>
<tr>
</table>

#### With Data Augmentation
<table>
<tr>
<th>Confusion Matrix </th>
<th>Normalized Confusion Matrix</th>
</tr>
<tr>
<th>
<img src="images/conf_matrix_no_norm_augm.png" alt="Confusion Matrix (Cross-validation)" width="400"/>
</th>
<th>
<img src="images/conf_matrix_normalized_row_augm.png" alt="Normalized Confusion Matrix (Cross-validation)" width="400"/>
</th>
<tr>
</table>


### TODO List
- [x] Automatic Download of dataset
- [x] CNN Architecture Definition
- [x] 3D Data Loader for Nifti files
- [x] Definition of loss functions
- [x] Training loop
- [x] Cross-validation on Train set
- [x] Command Line Interface for training 
- [x] Command Line Interface for validation 
- [x] 3D Data Augmentation 
- [ ] Tuning of Optimal Parameters for 3D Data Augmentation
- [ ] Validation on Test set

## Usage
Use ```python setup.py install``` for installing this package.
A complete run (dataset download, train, validation) of the package may be the following:
```console
git clone https://github.com/Nicolik/HippocampusSegmentationMRI.git
cd HippocampusSegmentationMRI
python setup.py install
python run/download.py
python run/train.py 
python run/validate.py
```
### Dataset
If you want to download the original dataset, run ```run/download.py```.
The syntax is as follows:
```console
python run/download.py --dir=path/to/dataset/dir
```
### Training
If you simply want to perform the training, run ```run/train.py```.
The syntax is as follows:
```console
python run/train.py --epochs=NUM_EPOCHS --batch=BATCH_SIZE --workers=NUM_WORKERS --lr=LR
```
If you want to edit the configuration, you can also modify the ```config/config.py``` file. 
In particular, consider the class ```SemSegMRIConfig```. 
If you want to play with data augmentation (built with ```torchio```), 
modify the ```config/augm.py``` file.

### Validation
If you want to perform the cross-validation, run ```run/validate.py``` or ```run/validate_torchio.py```.
The syntax is as follows:
```console
python run/validate.py --dir=path/to/logs/dir --write=WRITE --verbose=VERBOSE
```
```console
python run/validate_torchio.py --dir=path/to/logs/dir --verbose=VERBOSE
```
The former adopts a loop from scratch, whereas the latter exploits the DataLoader created upon ```torchio```. 

## Output Results
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