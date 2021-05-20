# Yoga class analysis
 ## About this project
This project outcome belongs to **Bram De Vroey**, **Joren Vervoort** and **Gülce Padem** who are currently junior Data Scientists/AI Operators in making at BeCode's Theano 2.27 promotion.

**Repository:** Yoga_Class_Analysis

**Type of Challenge:** Learning

**Duration:** 1 week

**Deadline:** 21/05/2021

**Deployment strategy:** Github page

**Contributers:** Gülce Padem, Bram De Vroey & Joren Vervoort

### Mission objectives

- Use of computer vision techniques for tracking poses on images and videos.
- Exploration of pre-trained models for pose tracking on live and streaming media.
- Deployment of models for end-customers.

### The Mission

The client is a wellbeing company based in NY city. They are currently providing yoga sessions online with trainers connected in real-time. This system allows the company to reach more people, facilitates the agenda of the clients, and increases the engagement on the sessions since the coaches are available to help.

Now, the job is becoming challenging for the coaches because it's hard for them to keep track of the progress of each one of the participants. They would like to evaluate if the people are doing correctly the poses and provide custom-made training plans but it's hard to do when hundreds are joining the same class.

Would you like to travel to NY? The company is asking for an AI developer with the skills to build an application able to track the poses done by the yoga practitioner, measure time, repetitions and evaluate if the poses are done correctly.

The company envisions an MVP where the customers receive a report of the yoga poses, which ones were done correctly, and metrics related to time and repetition. 

## Table of Contents

- Installation
- Repository
- Visual
- Pending things to do
- Acknowledgments
- Collaboration


## Important files in this repository

- Data_extraction.ipynb
- Extra Data.ipynb
- preprocessing.py
- full_data.csv
- model.py
- model.sav
- NN_classifier.ipynb
- NN_model (folder): 
- video_processing.py
- video.py
- webcam.py

## Installation


### Workflow

### Datascraping and preprocessing

#### Data_extraction.ipynb

In this jupyter notebook landmarks can be extracted (f.e.: nose, right_heel, left_hip, etc.) from given images of yoga poses from within the dataset. The XYZ coordinates and label of the yoga move are then stored in output.csv.

#### Extra Data.ipynb

To extend the trainingset for the models, extra images containing other people performing the yoga move, different camera angles had to be added. In this jupyter notebook extra images of the yoga poses can be automaticly downloaded. After this download, the images have to be checked for correctness by hand to ensure the right yoga move is performed within these pictures. Then the landmarks are extracted and added combined with output.csv into a new .csv file called full_data.csv.

#### preprocessing.py

#### full_data.csv

A .csv file containing the XYZ coordinates and labels of the performed yoga moves on the images in the dataset. 

### Model training

#### model.py
#### model.sav
#### NN_classifier.ipynb
#### NN_model

### Testing the model

#### video_processing.py
#### video.pywebcam.py
#### webcam.py

### Extra

#### timeline.png
#### README.md

## Visual



## Pending things to do

## Acknowlegdements

## Collaboration



# THANK YOU FOR READING!

