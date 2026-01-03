# load numpy
import numpy as np 

# load mnist database from tensorflow
from tensorflow.keras.datasets import mnist 


# sigmoid function makes the values between 0 and 1
def sigmoid(z):
    z = np.clip(z, -500, 500) #brings the extreme values to -500 and 500
    return 1/(1+np.exp(-z))


# defining the feedforward fn.
# it takes in a layer of input activations and returns activation of other layers
def feedForward(data):

    # data is a row vector (or vectors), so transpose to make it column vector.
    a1=data.T

    # pass through weights and biases to get second layer
    z2=weights1@a1 + bias1
    a2=sigmoid(z2)

    # similarly for other layers
    z3=weights2@a2 + bias2
    a3=sigmoid(z3)

    z4=weights3@a3 + bias3
    a4=sigmoid(z4)

    z5=weights4@a4 + bias4
    a5=sigmoid(z5)

    # return activation of all the layers
    return a1,a2,a3,a4,a5

# this fn finds the cost of the neural network
def findcost(output,label):

    l = len(label)
    expectedOutput = np.zeros((10, l), dtype=np.int32) #make numpy array of the same shape as output with 0s
    expectedOutput[label, np.arange(l)] = 1

    diff = output - expectedOutput # this will give the (a-y)
    cost = 0.5 * np.sum(np.square(diff))

    return diff,cost


# function to define the error in the final output layer
# inputs will be the (a-y) vector and the output of the final layer
def finalErrorOutput(diff,output):

    derivative = output * (1.0 - output) #find the derivative of sigma and plug output layers in it
    return derivative*diff # return the product of these 2 vectors which is our final error

# function to find the error for the other layers
# the weights from l-1 to l need to be given as they help us decide how to "blame" the neurons in the previous layer
def findError(currentLayerActivation,nextLayerError,connectingWeights):

    derivative=currentLayerActivation*(1-currentLayerActivation)
    blame = connectingWeights.T @ nextLayerError

    return derivative*blame

# function to save the weights and biases
def save_model(filename="brain.npz"):

    np.savez(filename, 
             w1=weights1, w2=weights2, w3=weights3, w4=weights4, 
             b1=bias1, b2=bias2, b3=bias3, b4=bias4)
    print(f"Saved the neural network")


# This downloads the files from the internet and puts them into 4 NumPy arrays
(x_train_raw, y_train), (x_test_raw, y_test) = mnist.load_data()

# the data is a array with 60k 28x28 matrices, we need to make it into 60k 784 element vectors
x_train_flat = x_train_raw.reshape(-1, 784)
x_test_flat = x_test_raw.reshape(-1, 784)

# normalising the data
# the dataset has 8 bit pixel values from 0 to 255
x_train_flat=x_train_flat.astype('float32') / 255.0 
x_test_flat=x_test_flat.astype('float32') / 255.0

#scaling for the random weights and biases
s=0.001 

#randomly setting the weights and biases
weights1=(np.random.rand(20,784)-0.5)*s
bias1=(np.random.rand(20,1)-0.5)*s
weights2=(np.random.rand(20,20)-0.5)*s
bias2=(np.random.rand(20,1)-0.5)*s
weights3=(np.random.rand(20,20)-0.5)*s
bias3=(np.random.rand(20,1)-0.5)*s
weights4=(np.random.rand(10,20)-0.5)*s
bias4=(np.random.rand(10,1)-0.5)*s

# function for batch gradient descent
def batchDescent(epochs,datasize,batchsize,learning_step):
    
    # to access the global variables
    global weights1, weights2, weights3, weights4
    global bias1, bias2, bias3, bias4 
    
    learning_step = learning_step/batchsize

    # main loop
    for epoch in range(epochs):

        print("Epoch:",epoch,"\n")

        # this loop will go through the dataset once
        for i in range(0,datasize,batchsize):
            
            # get the batch of images to train on
            x_batch = x_train_flat[i:i+batchsize]
            y_batch = y_train[i:i+batchsize]
            # these will be rows of column vectors which represent more than one image
            # instead of calculating individually, this will help calculate all at once

            # calculate the activation of layers and cost
            layer1,layer2,layer3,layer4,out=feedForward(x_batch)
            dOfCost,cost = findcost(out,y_batch)


            # calculate errors
            error5 = finalErrorOutput(dOfCost,out)
            error4 = findError(layer4,error5,weights4)
            error3 = findError(layer3,error4,weights3)
            error2 = findError(layer2,error3,weights2)

            # we only need 4 of these to update our parameters
            # logically it also makes sense as the first layer is input and we cannot affect the activations on it

            # finding the weight gradients
            wgrad4 = error5 @ layer4.T
            wgrad3 = error4 @ layer3.T
            wgrad2 = error3 @ layer2.T
            wgrad1 = error2 @ layer1.T

            # finding the bias gradients
            # we need to add up the columns
            # this happens automatically for the weight gradients
            bgrad4 = error5.sum(axis=1,keepdims=True) 
            bgrad3 = error4.sum(axis=1,keepdims=True) 
            bgrad2 = error3.sum(axis=1,keepdims=True) 
            bgrad1 = error2.sum(axis=1,keepdims=True) 
            
            # weight and bias gradients are added up so multiply by learning step to scale correctly

            # updating the weights
            weights4 = weights4 - (learning_step)*wgrad4
            weights3 = weights3 - (learning_step)*wgrad3
            weights2 = weights2 - (learning_step)*wgrad2
            weights1 = weights1 - (learning_step)*wgrad1

            # updating the biases
            bias4 = bias4 - (learning_step)*bgrad4
            bias3 = bias3 - (learning_step)*bgrad3
            bias2 = bias2 - (learning_step)*bgrad2
            bias1 = bias1 - (learning_step)*bgrad1
            

 
batchDescent(30,60000,32,6)

# save the model
save_model()

print("Use the neural network with \"usage.py\" ")
