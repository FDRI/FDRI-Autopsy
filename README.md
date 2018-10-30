# FDRI - Facial Detection and Recognition in Images module for Autopsy
FDRI-Autopsy V 1.0 - 17-09-2018

System requirements: 
 - Windows OS
 - CUDA capable GPU, cuDNN 5.0 support required

FDRI is a image analysis module that focus in finding human faces in images, 
as well finding images that contain a specific person. It provides this functionality’s appealing to AI Convolutional Neural Networks.
The executable is a implementation of facial detection and recognition with Dlib DNN(http://dlib.net/).

The facial recognition element is activated when selecting a folder with images from the person that 
the program should look for, it will look for the person and if it finds, marks it as interesting file hit.

All the detectors used can be found at: https://github.com/davisking/dlib-models

it can also be run as a standalone executable that requires .json file as paramenter, as example file is provided in sample folder.

# Authors:
 - Alexandre Frazão (ESTG / Politécnico de Leiria; Instituto de Telecomunicações - Portugal)
 - Patrício Domingues (CIIC / ESTG / Politécnico de Leiria; Instituto de Telecomunicações - Portugal)
 
Apache v2.0 version license is applied.
