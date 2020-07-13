# python run/download.py
# python run/download.py --dir=path/to/dataset/dir
# python run/download.py --dir=datasets

import argparse
import tarfile
import gdown
import os


def run(dataset_dir="datasets"):
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://drive.google.com/uc?id=1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C&export=download'
    output_tar = os.path.join(dataset_dir, 'Task04_Hippocampus.tar')
    if not os.path.exists(output_tar):
        gdown.download(url, output_tar, quiet=False)
    else:
        print("Output tar {} already exists!".format(output_tar))

    tar = tarfile.open(output_tar)
    tar.extractall(path=dataset_dir)
    tar.close()


############################
# MAIN
############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Dataset for Hippocampus Segmentation")
    parser.add_argument(
        "-D",
        "--dir",
        default="datasets", type=str,
        help="Local path to datasets dir"
    )

    args = parser.parse_args()
    run(dataset_dir=args.dir)
