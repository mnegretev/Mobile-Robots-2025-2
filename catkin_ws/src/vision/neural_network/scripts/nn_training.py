#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# TRAINING A NEURAL NETWORK
#
# Instructions:
# Complete the code to train a neural network for
# handwritten digits recognition.
#
import cv2
import sys
import random
import numpy
import rospy
import rospkg
import time

NAME = "JORGE EITHAN TREVIÑO SELLES"

class NeuralNetwork(object):
    def __init__(self, layers, weights=None, biases=None):
        #
        # The list 'layers' indicates the number of neurons in each layer.
        # Remember that the first layer indicates the dimension of the inputs and thus,
        # there is no bias vector fot the first layer.
        # For this practice, 'layers' should be something like [784, n2, n3, ..., nl, 10]
        # All weights and biases are initialized with random values. In each layer we have a matrix
        # of weights where row j contains all the weights of the j-th neuron in that layer. For this example,
        # the first matrix should be of order n2 x 784 and last matrix should be 10 x nl.
        #
        self.num_layers  = len(layers)
        self.layer_sizes = layers
        self.biases =[numpy.random.randn(y,1) for y in layers[1:]] if biases == None else biases
        self.weights=[numpy.random.randn(y,x) for x,y in zip(layers[:-1],layers[1:])] if weights==None else weights
        
    def forward(self, x):
        #
        # This function gets the output of the network when input is 'x'.
        #
        for i in range(len(self.biases)):
            u = numpy.dot(self.weights[i], x) + self.biases[i]
            x = 1.0 / (1.0 + numpy.exp(-u))  #output of the current layer is the input of the next one
        return x

    def forward_all_outputs(self, x):
        # Assign the first element of the list to the input
        ## This will be the first element of the list
        ## and the first layer of the network
        y = [x]
        for i in range(len(self.biases)):
            # Calculate the output of the layer
            u = numpy.dot(self.weights[i], x) + self.biases[i]
            # Apply the sigmoid function to the output
            x = 1.0 / (1.0 + numpy.exp(-u))
            # Append the output to the list
            y.append(x)
        
        return y

    def backpropagate(self, x, t):
        y = self.forward_all_outputs(x)
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        
        # Calculate delta for the output layer
        delta = (y[-1] - t) * y[-1] * (1 - y[-1])
        # Calculate nabla_b and nabla_w for the output layer
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, y[-2].T)
        
        # Iterate over the layers from L-1 to 1
        for l in range(2, self.num_layers):
            delta = numpy.dot(self.weights[-l+1].T, delta) * y[-l] * (1 - y[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, y[-l-1].T)
            
        return nabla_w, nabla_b

    def update_with_batch(self, batch, eta):
        #
        # This function exectutes gradient descend for the subset of examples
        # given by 'batch' with learning rate 'eta'
        # 'batch' is a list of training examples [(x,y), ..., (x,y)]
        #
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        M = len(batch)
        for x,y in batch:
            if rospy.is_shutdown():
                break
            delta_nabla_w, delta_nabla_b = self.backpropagate(x,y)
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w-eta*nw/M for w,nw in zip(self.weights, nabla_w)]
        self.biases  = [b-eta*nb/M for b,nb in zip(self.biases , nabla_b)]
        return nabla_w, nabla_b

    def get_gradient_mag(self, nabla_w, nabla_b):
        mag_w = sum([numpy.sum(n) for n in [nw*nw for nw in nabla_w]])
        mag_b = sum([numpy.sum(b) for b in [nb*nb for nb in nabla_b]])
        return mag_w + mag_b

    def train_by_SGD(self, training_data, epochs, batch_size, eta):
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0,len(training_data), batch_size)]
            for batch in batches:
                if rospy.is_shutdown():
                    return
                nabla_w, nabla_b = self.update_with_batch(batch, eta)
                sys.stdout.write("\rGradient magnitude: %f            " % (self.get_gradient_mag(nabla_w, nabla_b)))
                sys.stdout.flush()
            print("Epoch: " + str(j))
    #
    ### END OF CLASS
    #


def load_dataset(folder):
    print("Loading data set from " + folder)
    if not folder.endswith("/"):
        folder += "/"
    training_dataset, training_labels, testing_dataset, testing_labels = [],[],[],[]
    for i in range(10):
        f_data = [c/255.0 for c in open(folder + "data" + str(i), "rb").read(784000)]
        images = [numpy.asarray(f_data[784*j:784*(j+1)]).reshape([784,1]) for j in range(1000)]
        label  = numpy.asarray([1 if i == j else 0 for j in range(10)]).reshape([10,1])
        training_dataset += images[0:len(images)//2]
        training_labels  += [label for j in range(len(images)//2)]
        testing_dataset  += images[len(images)//2:len(images)]
        testing_labels   += [label for j in range(len(images)//2)]
    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))

def main():
    print("TRAINING A NEURAL NETWORK - " + NAME)
    rospy.init_node("nn_training")
    rospack = rospkg.RosPack()
    dataset_folder = rospack.get_path("neural_network") + "/handwritten_digits/"
    epochs        = 3
    batch_size    = 10
    learning_rate = 3.0
    
    if rospy.has_param("~epochs"):
        epochs = rospy.get_param("~epochs")
    if rospy.has_param("~batch_size"):
        batch_size = rospy.get_param("~batch_size")
    if rospy.has_param("~learning_rate"):
        learning_rate = rospy.get_param("~learning_rate") 

    training_dataset, testing_dataset = load_dataset(dataset_folder)
    
    nn = NeuralNetwork([784,30,10])
    nn.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)
    
    print("\nPress key to test network or ESC to exit...")
    numpy.set_printoptions(formatter={'float_kind':"{:.3f}".format})
    cmd = cv2.waitKey(0)
    while cmd != 27 and not rospy.is_shutdown():
        img,label = testing_dataset[numpy.random.randint(0, 4999)]
        y = nn.forward(img).transpose()
        print("\nPerceptron output: " + str(y))
        print("Expected output  : "   + str(label.transpose()))
        print("Recognized digit : "   + str(numpy.argmax(y)))
        cv2.imshow("Digit", numpy.reshape(numpy.asarray(img, dtype="float32"), (28,28,1)))
        cmd = cv2.waitKey(0)
    

if __name__ == '__main__':
    main()
