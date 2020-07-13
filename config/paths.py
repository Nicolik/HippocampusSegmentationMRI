import os

logs_folder = "logs"
os.makedirs(logs_folder, exist_ok=True)

base_dataset_dir = os.path.join("datasets","Task04_Hippocampus")
train_images_folder = os.path.join(base_dataset_dir, "imagesTr")
train_labels_folder = os.path.join(base_dataset_dir, "labelsTr")
train_prediction_folder = os.path.join(base_dataset_dir, "predTr")
train_images = os.listdir(train_images_folder)
train_labels = os.listdir(train_labels_folder)

train_images = [train_image for train_image in train_images
                if train_image.endswith(".nii.gz") and not train_image.startswith('.')]
train_labels = [train_label for train_label in train_labels
                if train_label.endswith(".nii.gz") and not train_label.startswith('.')]
test_images_folder = os.path.join(base_dataset_dir, "imagesTs")
test_images = os.listdir(test_images_folder)
test_prediction_folder = os.path.join(base_dataset_dir, "predTs")

test_images = [test_image for test_image in test_images
               if test_image.endswith(".nii.gz") and not test_image.startswith('.')]

labels_names = {
   "0": "background",
   "1": "Anterior",
   "2": "Posterior"
 }
labels_names_list = [labels_names[el] for el in labels_names]