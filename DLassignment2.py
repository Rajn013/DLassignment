#!/usr/bin/env python
# coding: utf-8

# Describe the structure of an artificial neuron. How is it similar to a biological neuron? What are its main components?
# 

# Artificial neuron also known as perceptron is the basic unit of the neural network . in simple terms, it is mathematical function based on a model of biological neuron. it can also be seen as a simple logic gate with binary output. they are something also called perceptrons.
# 
# take input from the input layer 
# weights them separately and sums them up.
# pass this sum through a nonlinear function to produce output.
# 
# the perceptron consist of 4 part:
#     Input values or one input layer
#     weights and bais
#     activation function 
#     output layer

# What are the different types of activation functions popularly used? Explain each of them.
# 

# the different types of activation function popularly used:
#     
#     1. sigmoid function: - in an ANN the simgmoid function is a non linear AF used primarily in feedforward netural network. it is a differentiable real funtion , defined for real input values and containing positive derivatives everywhere with a specific degree of smoothness.
#     
#     2. Hyperbolic Tangent Function (Tanh):- the hyperbolic tangent function , a..k..a.., the tanh function , is another type of AF.it is a smoother , zero - centered function having a range between -1 to 1. as a result the output of the tanh function is represented.
#     
#     3.softmax Function :-  the softmax function is another typer of AF used in neural network to compute probabolity distribution from a vector of real number this function generates an output that ranges between values 0 and 1 and with the sum of probabilities being equal to 1. 
#     
#     4.softsign Function :- the softsing function is another AF that is used in neural network computing. Althrough it is primarily in regresiion computation . it is a quadration polynomial
#         
#     5. Rectified Linear Unit (ReLU) function:- one of the most popular AFs in DL models, the rectified Linear unit (ReLU) function , is a fast - learning AF that promises to deliver state - of - the - art performance with stellar result . Compared to other AFs like the Sigmoid and tanh function , the Rehu function offers much better performance and generalization in deep learnig
#         the ReLU function performs a threshold operation on each input element where all value less then zero are set to zero.
#                         

# Explain, in details, Rosenblatt’s perceptron model. How can a set of data be classified using a simple perceptron?

# Rosenblatt perceptron model is a simple nerual network used for binary classification . it contain of input nodes, connection weight, a weighted sum function , an activation function and output node.
# the model classifies data point by calculating a weighted sum of the input value and applying an activation function to determine the output , which represents the classification .the process involves initilizing the weight, defining the Activation function preparing the data iterating over the data point calculating the weighted sum applying the activation function and evaluating the result . this can be implemnted by defining the weights and data point creating the activation function and iterating over the data point to calculate the classification

# Use a simple perceptron with weights w0, w1, and w2 as −1, 2, and 1, respectively, to classify data points (3, 4); (5, 2); (1, −3); (−8, −3); (−3, 0).

# In[8]:


#the perceotron  model is a type of binary classifer that separartes data point into two categories based ona decision boundary. -

w0 = -1
w1 = 2
w2 = 1

data =  [(3, 4), (5, 2), (1,-3), (-8,-3), (-3, 0)]

def activation_function(weighted_sum):
    return 1 if weighted_sum >= 0else 0

for point in data:
    x1, x2 = point
    
    weighted_sum = w0 + w1 * x1 + w2 * x2
    
    classification = activation_function(weighted_sum)
    
    print(f"data {point} is classified as  {classification}")


# Explain the basic structure of a multi-layer perceptron. Explain how it can solve the XOR problem.
# 

# a multi l;ayer perceptron (MLP) is a type of neural network that consist of multiple layers of interconnected nodes it has an input layer one more hidden layer and an output layer the information flow forward through the network from the input layer to the output layer with each layer performing calculation and passing the result to the next layer. the weight between the nodes are adjusted during training during training to learn the relationships in the data.
# 
# the XOR problem is a classic example that demondtrates the limitation of a single-layer perceptron which can only classify linearly separable data. the XOR problem invovle two input value(0 or 1) and a target output that is 1 only when the input are different. a single layer preceptron cannot find a linear decision boundary to correctly classify the XOR problem.
# MLP with a hidden layer can solve the XOR problem. The hidden layer allows for non-linear transformations of the input data, enabling the network to learn complex patterns. By combining multiple perceptrons and introducing non-linear activation functions, the MLP can learn to classify the XOR problem accurately

# What is artificial neural network (ANN)? Explain some of the salient highlights in the different architectural options for ANN.
# 

# Feedforward Neural Networks (FNNs): Information flows in one direction, from the input layer to the output layer, without cycles or feedback connections.
# 
# Recurrent Neural Networks (RNNs): They allow feedback connections, enabling information to be fed back into the network. RNNs are suitable for sequence data and tasks that require memory.
# 
# Convolutional Neural Networks (CNNs): Designed specifically for processing grid-like data, such as images. CNNs utilize convolutional layers to extract local patterns and hierarchical features.
# 
# Long Short-Term Memory Networks (LSTMs): A type of RNN architecture that overcomes the limitations of traditional RNNs in capturing long-term dependencies. LSTMs are well-suited for tasks involving sequential or time-series data.
# 
# Generative Adversarial Networks (GANs): Consist of two neural networks, a generator and a discriminator, which compete against each other. GANs are used for tasks like image generation and unsupervised learning.
# 
# Autoencoders: Neural networks designed to learn efficient representations of the input data. They consist of an encoder that compresses the input and a decoder that reconstructs it. Autoencoders are used for dimensionality reduction and anomaly detection.

# Explain the learning process of an ANN. Explain, with example, the challenge in assigning synaptic weights for the interconnection between neurons? How can this challenge be addressed?
# 

# the learning process of an artificial neural network (ANN) involves adjusting the synaptic weights between neurons to optimize the network's performance. Backpropagation is a common learning algorithm that involves forward propagation of input data, calculation of error, and backward propagation of the error to update the weights. Assigning synaptic weights is challenging because there is no direct formula for determining the "correct" weights. However, this challenge can be addressed by using iterative training with backpropagation, which adjusts the weights based on the error until the network's performance improves. The process involves presenting training data, calculating error, propagating it backward, and updating weights until satisfactory performance is achieved.

# Explain, in details, the backpropagation algorithm. What are the limitations of this algorithm?
# 

# The backpropagation algorithm is used in artificial neural networks to adjust the weights of connections between neurons. It involves two phases: forward propagation, where input data is processed and an error is calculated, and backward propagation, where the error is propagated back through the network to update the weights. The algorithm aims to minimize the error by adjusting the weights using gradient information. However, it has some limitations, including the potential to get stuck in local minima, overfitting, vanishing or exploding gradients, and sensitivity to hyperparameters. Despite these limitations, backpropagation remains a widely used and effective learning algorithm for training neural networks.
# 

# Describe, in details, the process of adjusting the interconnection weights in a multi-layer neural network.
# 

# The process of adjusting the interconnection weights in a multi-layer neural network involves initializing the weights, performing forward propagation to obtain network outputs, calculating the error between the outputs and the desired targets, propagating the error backward through the network, and updating the weights using an optimization algorithm like gradient descent. This process is repeated iteratively for multiple input samples until the network's performance improves. Python can be used to implement this process by defining the network architecture, activation functions, and applying the backpropagation algorithm to adjust the weights.

# What are the steps in the backpropagation algorithm? Why a multi-layer neural network is required?
# 

# the steps in the backpropagation algorithm are as follows:
# 
# Initialize the weights of the neural network randomly.
# Perform forward propagation to compute the outputs of each neuron.
# Calculate the error by comparing the network's output with the desired output.
# Backpropagate the error through the network to compute the gradients of the weights.
# Update the weights in the opposite direction of the gradients using an optimization algorithm like gradient descent.
# Repeat steps 2-5 for multiple input samples and iterations until the network's performance improves.

# short note :-
# 
# Artificial neuron
# Multi-layer perceptron
# Deep learning
# Learning rate
# 

# Artificial Neuron: A computational unit in an artificial neural network that applies weights to input signals, passes them through an activation function, and produces an output.
# 
# 
# Multi-layer Perceptron: A type of feedforward neural network consisting of multiple layers of interconnected artificial neurons. It includes an input layer, one or more hidden layers, and an output layer, enabling it to model complex relationships.
# 
# 
# Deep Learning: A subfield of machine learning that focuses on training neural networks with multiple layers. Deep learning models, such as deep neural networks, learn hierarchical representations of data and have achieved impressive performance in various domains.
# 
# 
# Learning Rate: A hyperparameter that determines the step size at which neural network weights are updated during training. It affects the convergence speed and stability of the training process, requiring careful selection for effective optimization.

# Activation function vs threshold function
# Step function vs sigmoid function
# Single layer vs multi-layer perceptron
# 

# Activation function vs Threshold function:
# 
# Activation function: It is a function applied to the output of a neuron in a neural network to introduce non-linearity. It allows the network to learn complex relationships between inputs and outputs. Examples include sigmoid, ReLU, and tanh functions.
# Threshold function: It is a type of activation function that produces binary outputs based on a predefined threshold. The output is either 0 or 1, depending on whether the input is above or below the threshold. It is commonly used in binary classification problems.
# 
#     
# Step function vs Sigmoid function:
# 
# Step function: It is a type of threshold function that produces a binary output based on whether the input is above or below a specific threshold. The output abruptly changes from one value to another. It is often used in simple binary classification problems.
# Sigmoid function: It is a type of activation function that maps the input to a value between 0 and 1, producing a smooth, S-shaped curve. It is commonly used in neural networks for binary classification and continuous prediction tasks. The sigmoid function provides a smooth transition and allows for the calculation of gradients during backpropagation.
# 
# Single layer vs Multi-layer perceptron:
# 
# Single layer perceptron: It is the simplest form of a neural network, consisting of a single layer of neurons. It can only learn linearly separable patterns and is limited in its ability to solve complex problems. It is often used for binary classification tasks.
# Multi-layer perceptron: It is a type of neural network with multiple layers of interconnected neurons. It includes an input layer, one or more hidden layers, and an output layer. The hidden layers enable the network to learn complex non-linear patterns and solve more intricate problems. It is widely used for various tasks such as image recognition, natural language processing, and regression.

# In[ ]:




