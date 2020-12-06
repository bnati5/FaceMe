# FaceMe
Identify if a person is wearing a face mask or whether the person is wearing it properly using Computer vision and deep learning.  


![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/bnati5/FaceMe/issues)
[![Forks](https://img.shields.io/github/forks/bnati5/FaceMe.svg?logo=github)](https://github.com/bnati5/FaceMe/network/members)
[![Stargazers](https://img.shields.io/github/stars/bnati5/FaceMe.svg?logo=github)](https://github.com/bnati5/FaceMe/stargazers)
[![Issues](https://img.shields.io/github/issues/bnati5/FaceMe.svg?logo=github)](https://github.com/chandrikadeb7/Face-Mask-Detection/issues)
[![MIT License](https://img.shields.io/github/license/bnati5/FaceMe.svg?style=flat-square)](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/LICENSE)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/nathanael-alemayehu-a77265191/)
  
For the live Demo!
[Click Here!](https://bnati5.github.io/FaceMe/)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Live Demo](images/demo.gif)


### The **[Dataset](https://www.kaggle.com/netflix-inc/netflix-prize-data)**
Masks play a crucial role in protecting the health of individuals against respiratory diseases, as is one of the few precautions available for COVID-19 in the absence of immunization. With this dataset, it is possible to create a model to detect people wearing masks, not wearing them, or wearing masks improperly.  
This dataset contains 853 images belonging to the 3 classes, as well as their bounding boxes in the PASCAL VOC format.  
The classes are:

* With mask
* Without mask
* Mask worn incorrectly


## Directory structure

            Face Mask Detection
            ├───data/
            │    ├───annotations/
            │    └───images/
            ├───notebooks/
            │    ├───Data-exploration-and-preprocessing.ipynb
            │    ├───FaceMe-MTCNN-face-detection.ipynb
            │    ├───FaceMe-ultra-light-face-detection.ipynb
            │    └───Traning-model.ipynb
            ├───modeljs/
            │    ├───model.json
            ...

## Workflow  

* Base model: InceptionV3 with imagenet weights.  
* Face detector: MTCNN  
* AVG FPS: 2.4 
* Model accuracy: 97%  

![Workflow](images/workflow.png)
