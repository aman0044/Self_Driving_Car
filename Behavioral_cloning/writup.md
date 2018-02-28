
# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed




I have used Nvidia model which is having 5 covolutional layer with 3 fully connected layers.All the details are mentioned below in point "2. Final Model Architecture".

 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting with keep probability of .40 .I have applied data augmentation to collect more data. The model was trained and validated on different data sets of images from center ,left and right angles of car to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.I had run my model for 6 epochs and batch size 64

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and runing vehicle in opposite direction .Moreover to i have augment the images so generate mroe variety of data. 

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to lower the training and validation loss and try to maintain the streeing angle such that car will run on the center of the lane

My first step was to use a convolution neural network model similar to the Nvidia model I thought this model might be appropriate because of the research involved from the nvidia people and this model has showed great results in their research paper.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set as well as validation set but while testing the car was not able to take few steep turns.This might be due to overfitting of data. 

To combat the overfitting, I modified the model which is mentioned in final model and i have augmented the images by flipping every image vertically ,cropping image from top and bottom and normalising image.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.



#### 2. Final Model Architecture

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)



____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param       Connected to                     

lambda_2 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_2 (Cropping2D)        (None, 80, 320, 3)    0           lambda_2[0][0]                   
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 38, 158, 24)   1824        cropping2d_2[0][0]               
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 17, 77, 36)    21636       convolution2d_6[0][0]            
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 7, 37, 48)     43248       convolution2d_7[0][0]            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 7, 37, 48)     0           convolution2d_8[0][0]            
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 5, 35, 64)     27712       dropout_2[0][0]                  
____________________________________________________________________________________________________
convolution2d_10 (Convolution2D) (None, 3, 33, 64)     36928       convolution2d_9[0][0]            
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 6336)          0           convolution2d_10[0][0]           
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 100)           633700      flatten_2[0][0]                  
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 50)            5050        dense_4[0][0]                    
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 1)             51          dense_5[0][0]                    

Total params: 770,149
Trainable params: 770,149
Non-trainable params: 0
____________________________________________________________________________________________________
 

I have used Nvidia model which is having 5 covolutional layer with 3 fully connected layers  and to optimise it for overfitting i have intoduced dropout of keep probability 0.4 after 3rd convolutional layer.The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving.
Here is an example image of center lane driving:

![center_2016_12_01_13_30_48_404.jpg](attachment:center_2016_12_01_13_30_48_404.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to 
recover from the conditions when it is going out of the track.
These images show what a recovery looks like starting from  right to center :

![center_2018_02_28_12_25_07_671.jpg](attachment:center_2018_02_28_12_25_07_671.jpg)
![center_2018_02_28_12_25_11_259.jpg](attachment:center_2018_02_28_12_25_11_259.jpg)
![center_2018_02_28_12_25_13_743.jpg](attachment:center_2018_02_28_12_25_13_743.jpg)


To augment the data sat, I also flipped images and angles thinking that this would increase my number of training data and inturn decrease overfitting. For example, here is an image that has then been flipped:
![center_2016_12_01_13_31_14_702.jpg](attachment:center_2016_12_01_13_31_14_702.jpg)
![image.jpg](attachment:image.jpg)

After the collection process, I had 3 number of data points. I then preprocessed this data by normalising it by lambda function and the  cropping the top 70 row pixels and 25 bottom pixels.

I finally randomly shuffled the data set. 

I used this training data for training the model.
The validation set helped determine if the model was over or under fitting.
The ideal number of epochs was 6 .
I used an adam optimizer so that manually training the learning rate wasn't necessary.

