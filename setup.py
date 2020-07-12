import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Hippocampus-Segmentation-VNet-Torch", # Replace with your own username
    version="0.0.1",
    author="Nicola Altini",
    author_email="nicola.altini@poliba.it",
    description="Hippocampus Segmentation from MRI using 3D Convolutional Neural Networks in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nicolik/HippocampusSegmentationMRI",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    test_suite='tests',
)