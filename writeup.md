# **Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
  
* The size of test set is 12630

* The shape of a traffic sign image is (32, 32, 3)
  
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The code for this step is contained in the third code cell of the IPython notebook.  


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Although colors in the traffic sign are important in real world for
people to recoganize different signs, traffic signs are also different
in their shapes and contents. We can ignore colors in this problem
because signs in our training set are differentiable from their
contents and shapes, and the network seems having no problem to learn
just from shapes.

Therefore, My preprocessing phase normalizes images from [0, 255] to
[0, 1], and grayscales it. You can see the grayscale effects.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The train, valid and test data are prepreocessed in cell 9. I use
cross validation to split training data. The code to split the data
is in function `train` (see cell 15).

To cross validate my model, I randomly split the given training sets
into training set and validation set. I preserved 10% data for
validation. `sklearn` has the handy tool `train_test_split` to do the
work.


The code is in function `classifier` .

I adapted LeNet architecture: Two convolutional layers followed by one
flatten layer, drop out layer, and three fully connected linear
layers.

1. convolution 1: 32x32x1  -> 28x28x12 -> relu -> 14x14x12 (pooling)
2. convolution 2: 14x14x12 -> 10x10x25 -> relu -> 5x5x25   (pooling)
3.       flatten: 5x5x25   -> 625
4.      drop out: 625      -> 625
5.        linear: 625      -> 300
6.        linear: 300      -> 150
7.        linear: 150      -> 43


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


I train the model in 10 iterations (epochs), and each iteration is
trained with 64 batch size. Adam optimizer is used with learning rate
0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.934
* test set accuracy of 0.927

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  *  I used a very similar architecture to the paper offered by the instructors.  I used it because they got such a good score the answer was given through it.
* What were some problems with the initial architecture?
  *  The first issue was lack of data for some images and the last was lack of knowledge of all the parameters.  After I fixed those issues the LeNet model given worked pretty well with the defaults.  I still couldn't break 98% very easily until I added another convolution.  After that it was much faster at reaching higher accuracy scores.
* How was the architecture adjusted and why was it adjusted?
  * Past what was said in the previous question, I didn't alter much past adding a couple dropouts with a 50% probability.
* Which parameters were tuned? How were they adjusted and why?
  * Epoch, learning rate, batch size, and drop out probability were all parameters tuned along with the number of random modifications to generate more image data was tuned.  For Epoch the main reason I tuned this was after I started to get better accuracy early on I lowered the number once I had confidence I could reach my accuracy goals.  The batch size I increased only slightly since starting once I increased the dataset size.  The learning rate I think could of been left at .001 which is as I am told a normal starting point, but I just wanted to try something different so .00097 was used.  I think it mattered little.  The dropout probability mattered a lot early on, but after awhile I set it to 50% and just left it.  The biggest thing that effected my accuracy was the data images generated with random modifications.  This would turn my accuracy from 1-10 epochs from 40% to 60% max to 70% to 90% within the first few evaluations. Increasing the dataset in the correct places really improved the max accuracy as well.
* What are some of the important design choices and why were they chosen? I think I could go over this project for another week and keep on learning.  I think this is a good question and I could still learn more about that.  I think the most important thing I learned was having a more uniform dataset along with enough convolutions to capture features will greatly improve speed of training and accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I want to see how the classifier performs on similar signs. The
General Caution and Traffic signals: they both look like a vertical bar
(see the visualization) when grayscaled. And pedestrains and child
crossing look similar in low resolution.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The
result is explained and virtualized in detail in cell 28.

The accuracy on the new traffic signs is 63.6%, while it was 93% on
the test set. This is a sign of underfitting. By looking at the
virtualized result, I think this can be addressed by using more image
preprocessing techniques on the training set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

In the submitted version, the model can correctly guess 7 out of 11
signs. The accuracy is 63.6%. However, it can sometimes predict
correctly 10 out of 11 images.

