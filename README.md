# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/architecture.PNG "Architecture"
[image2]: ./images/mse.png "MSE"
[image3]: ./images/camera.PNG "Camera Image"
[image4]: ./images/flip.PNG "Fliped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The architecture of the model can be found in the below summary.
![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.(model.py line 84). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 92).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used pictures taken from the left, center, and right cameras so that the vehicle can be driven in the middle of the lane. In addition, data on driving rotating clockwise and counterclockwise were also utilized.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The first step was to use a convolutional neural network model, as shown in the model architecture described above. Input the image and output (x), steering angle (y). The images and steering angle produced by the center camera were used as they were, but 0.2 was added as a correction value to the steering angle produced by the camera on the left, and the correction value was subtracted in the opposite case (model.py line 37-42).
![alt text][image3]
![alt text][image4]

The training and validation sets were divided by 0.8 and 0.2 respectively. The training model can be seen in the architecture.

To prevent overfitting, a dropout layer was added with a 0.5 dropout ratio, and a flattening layer was added (model.py line 84).

In this project, special consideration was given to pre-processing the photographic data, and it was successful in safe autonomous driving.
In order to drive safely on the left and right corners, the picture was flipped left and right. In this case, change the sign of the steering angle (model.py line 43). 
Also, when training after changing the image to rgb, the vehicle drove more stably (model.py line 34). 
Images were cropped the upper and lower parts of the unnecessary parts (model.py line 76).

At the end of the process, the vehicle can autonomously drive the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

All these data was shuffled randlomly and used for training the model with five epochs. The following picture shows the training:

![alt text][image2]
