# DLR-Detection
## Introduction
This repository was developed as a project of the Dienstleistungszentrum Ländlicher Raum Rheinland Pfalz. The goal was to develop an algorithm that can automatically detect apples and different stages of apple flowers in images. It is designed to work with an entire row of trees. Given images of the row it can produce an approximate number of apples and flowers. This can help farmers get an approximate yield prediction and help them with decisions like if and where they need to thin out flowers.

## How it works
The program works with a sequence of images of a row of apple trees. This sequence should contain one image for every tree in the row. If your images are not filtered this way you can use the program to automatically filter the images to have one image per tree. This automatic filtering needs every image to have a GPS coordinate and builds a list of equally spaced coordinates between the coordinates of the first and last trees. The number of coordinates in the list is equal to the number of trees in the row. It then assigns one image to each coordinate based on the GPS coordinate of the image. 

<br>

The images are fed to a YOLOv5 model. This model has been pre-trained to detect trees, apples, apple flowers, and apple flower buds. The pre-trained weights are contained in this repository. The results contain an approximate number of the detected classes for the entire row and each image.

<br>

In addition to a single image per tree, the program also works for a three-camera setup. This three-camera setup is part of the sensor platform used in the project to capture data. It consists of three vertically aligned cameras at different heights. This results in three images per tree at different heights. Detections from the images are added together taking all of the detections from the top and bottom image and 10% of the detections from the middle image. The image from the top camera can also be cropped to reduce overlap between the images. The automatic filter also works for three cameras.

## Usage
Open the file named "DLRDetectionNotebook.ipynb" located in the main folder. In the notebook, all necessary options can be set. It also contains explanations for each option. After all options are set simply run the notebook. All results will be saved in the results folder.

# Installation
Anaconda package manager is required to install the dependicies via the yml file.
1. Clone this repository:

2. Create a new Conda environment using the yml file
```
conda env create -f environment.yml
```
In addition, some program to run the main Jupyter Notebook is needed.
