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

[image8]: ./examples/center.png "center"
[image9]: ./examples/left.png "left"
[image10]: ./examples/right.png "right"

[image11]: ./examples/center_flipped.png "center flipped"
[image12]: ./examples/left_flipped.png "left flipped"
[image13]: ./examples/right_flipped.png "right flipped"

[image14]: ./examples/mse.png "mse"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Also of note is that I am using Keras 2.0.6 which changed the format of how convolutional 2d layers work. The main change was that rather than passing `samples_per_epoch` I instead had to pass `steps_per_epoch` which is basically samples_per_epoch/batch_size. The same applied to `nb_val_steps` and `val_samples`.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network having 3 convolution layers with 5x5 filters and another 2 convolution layers with 3x3 filters. This is followed by 3 fully-connected layers and a final layer for the steering angle predicted value (model.py lines 77-82).

The model includes RELU layers to introduce nonlinearity (code line 77, 78, 79, 81, 82), and the data is normalized in the model using a Keras lambda layer (code line 74).

I added a Cropping2D layer that helped to reduce training "noise" by excluding the area of the images that consisted of the bulk of the car and sky (code line 75)."

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 86 and 88). 

The model was trained and validated on the Udacity-provided data as well as data I gathered and augmented. The model was tested by running it through the simulator both two different tracks and ensuring that the vehicle could stay on the tracks.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving in reverse (on track 1) and using an analog controller to get refined driving angles.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My initial strategy was to create a simple regression network that inputted an image and outputted a stearing angle with mean square error (mse) for my loss function and Adam as my optimizer. Once I verified that I was able to create a model that would drive the simulator I proceeded to implement my current model, which is based on a convultion neural network similar to that found in this Nvidia paper on end to end learning https://arxiv.org/pdf/1604.07316.pdf.

I felt this choice of model was appropriate because it directly applied to our use case of training a car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model in include dropout layers of 0.5 (model.py lines 86 and 88).

I also attempted to increase the amount of samples that I had to train with. Because track 1 is fairly simple with lots of left turns, I opted to attemt to augment the data rather than do more laps. This involved including data recorded from both the left and right cameras and then modifying the angles associated with them to be similar to that of the center camera. I settled on a correct value of 0.28 that I added to the left camera's angle and subtracted from the right camera's angle.

Finally I took all three camera images and flipped them then added them to my sample data.

The final step was to run the simulator to see how well the car was driving around track one. 

After about two center lane driving with an analog controller on track 2 I retrained my model and got results that were perfect up to speeds of about 25 mph. There was one particular spot on track 2 where at maximum speed (30mph) the car would collide with a wall during a sharp turn. To counter this I recorded myself driving towards this area a few times and then making a sharp turn before the wall.

At the end of the process, the vehicle is able to drive autonomously around both tracks 1 and 2 at 30 mph without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 73-90) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture using keras' model.summary()

Layer (type)                 Output Shape              Param #   

lambda_4 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_4 (Cropping2D)    (None, 75, 320, 3)        0         
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 36, 158, 24)       1824      
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 16, 77, 36)        21636     
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 6, 37, 48)         43248     
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 4, 35, 64)         27712     
_________________________________________________________________
conv2d_20 (Conv2D)           (None, 2, 33, 64)         36928     
_________________________________________________________________
flatten_4 (Flatten)          (None, 4224)              0         
_________________________________________________________________
dense_13 (Dense)             (None, 100)               422500    
_________________________________________________________________
dropout_7 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_14 (Dense)             (None, 50)                5050      
_________________________________________________________________
dropout_8 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_15 (Dense)             (None, 10)                510       
_________________________________________________________________
dense_16 (Dense)             (None, 1)                 11        

Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0

Additional visualizations of this network can be found in NVIDIA's paper: https://arxiv.org/pdf/1604.07316.pdf

####3. Creation of the Training Set & Training Process

I did my initial training and augmentation using Udacity's provided sample data. I added additional data to it by doing some test laps using an analog controller to get more precise angles around turns. This allowed my model to train with good driving angles through bigger turns which reduced the need to potentially train recovery turns. 

Here are some examples of some of the images that the simulator records for the left, center and right cameras.

![alt text][image8] ![alt text][image9] ![alt text][image10]

I also recorded myself swaying back and forth making angles sharper than I would normally take during "good" driving to inform the model on how to recover.

Then I repeated this process on track two in order to get more data points.

The first course overwhelmingly was made up of left turns which initially lead to a model that favored making left turns. I combatted this overfitting of the data by augmenting my data by flipping all the images and adding them to my samples. 

Here are the above images flipped:

![alt text][image12] ![alt text][image11] ![alt text][image13]


The sample data contained 8036 samples. Through my own collection I included an additional 9869 samples. Using both the sample data and the data that I collected on both tracks 1 and 2 I had a total of 17905 samples. 

After adding both the left and right cameras as well as the flipped images for all three my total samples increased to 107430.

Number of samples     :  107430
Number of train samples     :  85944
Number of validation samples:  21486

I then shuffled the samples then split the them 80% into training and 20% into validation.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the following chart. Both training loss and validation decrease until about epoch 5 where validation loss begins to increase. I tweaked my model until both training and validation loss was around 0.2, further reductions in loss didn't seem to have a meaningful impact.

![alt text][image14]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

Due to shear amount of data that I had to work with I used Keras fit_generator to batch loading and augmentation of training images. Training was done on an nvidia 1080ti.


Epoch 1/5
2686/2685 [==============================] - 173s - loss: 0.0462 - val_loss: 0.0353
Epoch 2/5
2686/2685 [==============================] - 105s - loss: 0.0346 - val_loss: 0.0316
Epoch 3/5
2686/2685 [==============================] - 106s - loss: 0.0305 - val_loss: 0.0291
Epoch 4/5
2686/2685 [==============================] - 104s - loss: 0.0276 - val_loss: 0.0276
Epoch 5/5
2686/2685 [==============================] - 105s - loss: 0.0253 - val_loss: 0.0259


