# Traffic Sign Recognition Project 


The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

I will discuss those steps in the current writeup following primarily the discussion of the template notebook. The main material has been developed in the [notebook](./Solution-Traffic_Sign_Classifier.ipynb), for which a respective downloaded HTML is also in the repository at the linked [location](./Solution-Traffic_Sign_Classifier.html).

---

[traffic_signs]: ./docs/gallery.jpg "Traffic signs"
[rel_frequency]: ./docs/ref_freq.jpg "Empirical distribution"
[learning]: ./docs/learning.jpg "Learning history"
[new_images]: ./docs/prediction_new_images.jpg "Traffic Sign 1"


# Description

## 1. Environment
Local environment was created via the modified ``environment_mod_gpu.yml``. Note that ``cudnn=6`` is still missing for proper usage of TensorFlow GPU 1.3, which does not seem to be provided anywhere on the common conda channels. Thus, download directly from NVIDIA and extract its content to the ``Library`` folder of the environment folder.

## 2. Dataset

This section includes the following tasks:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set

The traffic sign data has been provided via the following [location](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip), which contains directly named pickle files of training, validation and test dataset. 

The traffic signs are colored images of size ``32x32`` and we have in total 43 different sign classes. The training set has 34799, the validation set has 4410 and the test set has 12630 samples. 

An example of a random image per class id from the training set is shown in the figure below. A mapping from class id to name is provided in the ``signnames.csv`` file. As you can observe, some cases are really challenging already with the human eye, e.g., id number 5 or 23, and the quality of some images is really poor.

![][traffic_signs]

Additionally, let us check and visualize the class distribution. The following figure shows the relative distribution over the class for the training and validation set. While the classes are not unbiased, there are at least no classes grossly under- or over-represented. Additionally, the distribution between training and validation set is similar.

![][rel_frequency]


## 3. Neural network

* Design, train and test a model architecture


The biggest challenge was to employ native TensorFlow, its old version and then even using the low level API. However, I thought it might be a good exercise for getting a more detailed impression of what it was like some years ago.

Let me comment on the choices:

- Preprocessing: Normalize image from ``uint8`` to ``float32`` in the range [-1, 1]. No grayscale conversion was performed because color is a good additional feature of identifying traffic signs.
- Network architecture: The architecture is very similar to LeNet5 with the appropriate in- and output adjustments and more neurons. Let me include the ``create_model`` function here, because it is the central part of the work and I commented the structure; apparently Python code in itself has a high level of readability: 
```python
# model architecture


def create_model(x):

    # block 1: 32, 3x3 + 1(S) conv, 2x2 + 2(S) max-pool
    # shapes: (32, 32, 3) -> (30, 30, 32) -> (15, 15, 32)
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, CHANNELS, 32), mean=MU, stddev=SIGMA))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # block 2: 64, 3x3 + 1(S) conv, 2x2 + 2(S) max-pool
    # shapes: (15, 15, 32) -> (13, 13, 64) -> (6, 6, 64)
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean=MU, stddev=SIGMA))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # block 3: flatten, fully connected 256
    # shapes: (6, 6, 64) -> (2304,) -> (256,)    
    fc1 = tf.reshape(conv2, [-1, 2304])
    fc1_W = tf.Variable(tf.truncated_normal(shape=(2304, 256), mean=MU, stddev=SIGMA))
    fc1_b = tf.Variable(tf.zeros(256))
    fc1 = tf.matmul(fc1, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)
    
    # block 4: fully connected 256
    # shapes: (256,) -> (256,)
    fc2_W = tf.Variable(tf.truncated_normal(shape=(256, 256), mean=MU, stddev=SIGMA))
    fc2_b = tf.Variable(tf.zeros(256))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)
    
    # logits: fully connected 43
    # shapes: (256,) -> (43,)
    fc3_W = tf.Variable(tf.truncated_normal(shape=(256, NUM_CLASSES), mean=MU, stddev=SIGMA))
    fc3_b = tf.Variable(tf.zeros(NUM_CLASSES))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
```
- Approach: Already LeNet5 trained for more epochs was giving a validation accuracy above 0.93. However with more neurons this was achieved even more consistently. Thus, I also did not pursue any further options like augmentation, regularization, etc. 
- Training: Finally I trained for 50 epochs using a batch size of 128 and the Adam optimizer with an initial learning rate of 1e-3. There is clear overfitting between training and validation, but the required accuracy on the validation set is surpassed consistently, shown by the dotted line on the right side: ![][learning]

After having trained and tuned the model, the evaluation on the test set delivers:

| Test Set         		|          | 
|:---------------------:|:--------:| 
| Cross entropy loss    | 0.3866   |
| Accuracy              | 95.30 %  |


## 4. New Images

The section covers:

* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

In the last part we were supposed to test the model on at least 5 new images and check and discuss its performance. 

The original images are uploaded to the folder [``new_images``](./new_images). I search on the web and was capturing roughly squared-sized images of traffic signs. Note after inital loading they are resized so that the network accepts the input. The result is shown in the following figure, showing the probabilities of the top-5 probabilities, the image and the manual label true class (title of the figure):

![][new_images]

Let me add the following discussion:
- The overall quality of the images is good, all traffic signs have been present in the dataset, and they are well captured from the front. I also picked a slightly snow covered traffic sign 'Beware of ice/snow' (last picture), for which I was interested to see if it is classified correctly while looking for images.
- All 5 out of 5 cases have been classified correctly in this example, but with only 5 images no statistical sound comparison to the test accuracy is possible in my opinion.
- From the top-5 prediction classes one observes that certain cases have been more difficult than others. While the 'No passing' sign has been more or less classified correctly to certainty, cases like the snow covered sign 'Beware of ice/snow' are not as certain. Here also a human might have some problems at the given resolution.

