# EXERCISE-POOLING-PERFOMANCE

## This Content was Created by Intel Edge AI for IoT Developers UDACITY Nanodegree. (Solution of the exercise and adaptation as a repository: Andrés R. Bucheli.)

Calculate the total number of FLOPS for a Deep Learning Model (Convolutional layers and average pooling layers).  Run the model and calculate Inference Time with Intel OpenVino Toolkit.

# Exercise: Pooling Performance

For this exercise, your first task will be to calculate the total number of FLOPs for the <code>pool_cnn</code> model given below. Your second task will be to run this model and measure the inference time.

![image](https://raw.githubusercontent.com/ARBUCHELI/EXERCISE-POOLING-PERFOMANCE/master/l3-slides-poolin.jpg)

# Task 1: Calculate Model FLOPs
## Layer 1: Conv2D
Input shape: 1x1x28x28
Kernel shape: 3x3
Number of kernels: 10
## Solution:
Output shape:
The shape for a single dimension will be = (28-3)+1 = 26
So our output shape will be 26x26
Because we have 10 kernels, our actual output shape will be 10x26x26
FLOPs: 10x26x26x3x3x1x2 = 121,680

## Layer 2: Average Pool 2D
Input Shape: 10x26x26
Kernel Shape: 2x2
## Solution:
Output Shape: 10x13x13
FLOPs: 13x13x2x2x10 = 6,760

## Layer 3: Conv2D
Input shape: 10x13x13
Kernel shape: 3x3
Number of kernels: 5
## Solution:
Output shape:
The shape for a single dimension will be = (13-3)+1 = 11
So our output shape will be 11x11
Because we have 5 kernels, our actual output shape will be 5x11x11
FLOPs: 5x11x11x3x3x10x2 = 108,900

## Layer 4: Fully Connected
Input shape: 11x11x5: 605
Output shape: 128
## Solution:
FLOPs: 605x128x2 = 154,880

## Layer 5: Fully Connected
Input Shape: 128
Output Shape: 10
## Solution:
FLOPs: 128x10x2 = 2560
Total FLOPs: 121680+6760+108900+154880+2560 = 394,780

## Task 2: Completing the Inference Pipeline
Your next task is to complete the <code>inference.py</code> python script on the right.

Remember to source the OpenVINO environment before running the script.

To run the <code>inference.py</code> file, you can use the command:
<code>python3 inference.py</code>

<strong>Note:</strong> You may get a warning about OpenVINO using a different Python version. You can ignore this warning, the inference should still run fine.

<pre><code>
from openvino.inference_engine import IENetwork, IECore

import numpy as np
import time

# Getting model bin and xml file
model_path='pool_cnn/pool_cnn'
model_weights=model_path+'.bin'
model_structure=model_path+'.xml'

# TODO: Load the model
# Use either IECore or IEPlugin API
model=IENetwork(model_structure, model_weights)

core = IECore()
net = core.load_network(network=model, device_name='CPU', num_requests=1)

input_name=next(iter(model.inputs))

# Reading and Preprocessing Image
input_img=np.load('image.npy')
input_img=input_img.reshape(1, 28, 28)


# TODO: Using the input image, run inference on the model for 10 iterations

input_dict={input_name:input_img}

start=time.time()
for _ in range(10):
    net.infer(input_dict)

# TODO: Finish the print statement
print("Time taken to run 10 iterations is: {} seconds".format(time.time()-start))
</code></pre>

## Solution of the exercise and adaptation as a Repository: Andrés R. Bucheli.
