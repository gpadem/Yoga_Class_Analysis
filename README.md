# Yoga class analysis

In this project, we use computer vision to classify yoga poses from video. We use [MediaPipe]() for the pose detection, and then machine learning or a neural network to classify the resulting data points to a small selection of yoga poses.

## Table of contents

 ## About this project
This project outcome belongs to **Bram De Vroey**, **Joren Vervoort** and **Gülce Padem** who are currently junior Data Scientists/AI Operators in making at BeCode's _Theano 2.27_ promotion.

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

## Important files in this repository
- **Exploration**:
    - [Data_extraction.ipynb](Data_extraction.ipynb)
    - [Extra Data.ipynb](./'Extra Data.ipynb')
    - [NN_classifier.ipynb](NN_classifier.ipynb)
    - webcam.py
    - video.py
- Usage scripts 
    - preprocessing.py
    - video_processing.py
- Training data
    - full_data.csv
- Trained models
    - NN_model (folder)
    - model.sav
    - model.py

## Installation & Use

All necessary packages are in [requirements.txt](requirements.txt), best make a virtual environment and install wtih 
```python 
pip install -r requirements.txt
```

To *use* the processing pipeline, you run **video_processing.py**, however this is very basic and not very well-tested.

```
python video_processing.py VIDEOFILE [rf|nn]
```


The main part is the function `process_video()` in _video_processing.py_, which can be used to convert a video to an annoted one, and also returns a dataframe with probability distribution of all poses for each analyzed frame.For all options, look in the docstrings of the python files.



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

#### Machine Learning
#### model.py
In this file a multiclass classifier (random forest) is trained and evalutated to label the 12 different yoga moves. 

#### model.sav
A file containing the trained machine learning model.

#### Neural Network
#### NN_classifier.ipynb

#### NN_model

### Testing the model

#### video_processing.py
#### video.py
With this file a video can be analyzed based on the two created models. Displaying the move with the highest probability.

#### webcam.py
With this file a live video can be analyzed based on the two created models. Displaying the move with the highest probability.

### Extra

#### timeline.png
This .png file represents the probabilities of all the different yoga moves during a 1 minute video.

#### README.md
A file to explain the files, approach, etc. of this project.

## Visual



## Pending things to do

## Acknowlegdements

## Collaboration



# THANK YOU FOR READING!

