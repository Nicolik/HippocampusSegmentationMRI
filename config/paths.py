import os

logs_folder = "logs"
base_dataset_dir = os.path.join("datasets","Task04_Hippocampus")
train_images_folder = os.path.join(base_dataset_dir, "imagesTr")
train_labels_folder = os.path.join(base_dataset_dir, "labelsTr")
train_prediction_folder = os.path.join(base_dataset_dir, "predTr")
train_images = os.listdir(train_images_folder)
train_labels = os.listdir(train_labels_folder)

train_images = [train_image for train_image in train_images
                if train_image.endswith(".nii.gz")]
train_labels = [train_label for train_label in train_labels
                if train_label.endswith(".nii.gz")]
test_images_folder = os.path.join(base_dataset_dir, "imagesTs")
test_images = os.listdir(test_images_folder)
test_prediction_folder = os.path.join(base_dataset_dir, "predTs")
