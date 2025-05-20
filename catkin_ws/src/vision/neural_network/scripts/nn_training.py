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
import time       # Import the time module
import csv        # Import the csv module

NAME = "Germán Zaír Romero Hernández" # Remember to change this to your full name

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
<<<<<<< HEAD

=======
        
>>>>>>> 03c75fe8058c1a1b796dc93635380a9cc4155fda
    def forward(self, x):
        #
        # This function gets the output of the network when input is 'x'.
        #
        for i in range(len(self.biases)):
            u = numpy.dot(self.weights[i], x) + self.biases[i]
            x = 1.0 / (1.0 + numpy.exp(-u))  #output of the current layer is the input of the next one
        return x

    def forward_all_outputs(self, x):
<<<<<<< HEAD
        y = [x] # Include input x as the first output.
=======
        y = []
>>>>>>> 03c75fe8058c1a1b796dc93635380a9cc4155fda
        #
        # TODO:
        # Write a function similar to 'forward' but instead of returning only the output layer,
        # return a list containing the output of each layer, from input to output.
        # Include input x as the first output.
        #
        a = x
        for b, w in zip(self.biases, self.weights):
            u = numpy.dot(w, a) + b
            a = 1.0 / (1.0 + numpy.exp(-u)) # Sigmoid activation
            y.append(a)
        return y

    def backpropagate(self, x, t):
        y = self.forward_all_outputs(x)
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        # TODO:
        # Return a tuple [nabla_w, nabla_b] containing the gradient of cost function C with respect to
        # each weight and bias of all the network. The gradient is calculated assuming only one training
        # example is given: the input 'x' and the corresponding label 'yt'.
        # nabla_w and nabla_b should have the same dimensions as the corresponding
        # self.weights and self.biases
        # You can calculate the gradient following these steps:
        #
        # Calculate delta for the output layer L: delta=(yL-yt)*yL*(1-yL)
        # nabla_b of output layer = delta
        # nabla_w of output layer = delta*yLpT where yLpT is the transpose of the ouput vector of layer L-1
        # FOR all layers 'l' from L-1 to input layer:
        #     delta = (WT * delta)*yl*(1 - yl)
        #     where 'WT' is the transpose of the matrix of weights of layer l+1 and 'yl' is the output of layer l
        #     nabla_b[-l] = delta
        #     nabla_w[-l] = delta*ylpT  where ylpT is the transpose of outputs vector of layer l-1
        #

        # Calculate delta for the output layer
        delta = (y[-1] - t) * y[-1] * (1 - y[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, y[-2].transpose())

        # Backpropagate the error
        for l in range(2, self.num_layers):
            z = y[-l]
            sp = z * (1 - z) # Derivative of sigmoid
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, y[-l-1].transpose())

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
                # print("\rGradient magnitude: %f           " % (self.get_gradient_mag(nabla_w, nabla_b)), end="") # Modified for better output in terminal
            print(f"Epoch {j+1}/{epochs} completed.") # Modified print to show progress per epoch
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
        testing_dataset   += images[len(images)//2:len(images)]
        testing_labels   += [label for j in range(len(images)//2)]
    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))

def main():
    print("TRAINING A NEURAL NETWORK - " + NAME)
    rospy.init_node("nn_training")
    rospack = rospkg.RosPack()
    dataset_folder = rospack.get_path("neural_network") + "/handwritten_digits/"

    # Define parameter values to iterate through
    epochs_list = [3, 10, 50, 100]     # Example values for epochs
    batch_size_list = [5, 10, 30, 100] # Example values for batch size
    # Corrected learning rate list - removed the extra dot
    learning_rate_list = [0.5, 1.0, 3.0, 10.0] # Example values for learning rate
    hidden_layer_sizes = [30, 50]      # Example values for hidden layer size

    results = []
    num_test_samples = 100 # Number of samples to use from the testing dataset for accuracy calculation

    training_dataset, testing_dataset = load_dataset(dataset_folder)
<<<<<<< HEAD

    # Ensure we have enough samples for testing
    if len(testing_dataset) < num_test_samples:
        print(f"Warning: Testing dataset has only {len(testing_dataset)} samples, less than the requested {num_test_samples}.")
        num_test_samples = len(testing_dataset)


    for epochs in epochs_list:
        for batch_size in batch_size_list:
            for learning_rate in learning_rate_list:
                for hidden_size in hidden_layer_sizes:
                    if rospy.is_shutdown():
                        break

                    print(f"\n--- Running Experiment ---")
                    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Hidden Layer Size: {hidden_size}")

                    # Initialize and train the neural network
                    # A new network is created for each parameter combination
                    nn = NeuralNetwork([784, hidden_size, 10])

                    start_time = time.time()
                    nn.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)
                    end_time = time.time()
                    training_time = end_time - start_time

                    # Calculate accuracy using a limited number of test samples
                    correct_predictions = 0
                    # Iterate only over the first num_test_samples
                    for img, label in testing_dataset[:num_test_samples]:
                        if rospy.is_shutdown():
                            break

                        # Get the network output for the current image
                        y = nn.forward(img).transpose()

                        # Determine the predicted digit (the one with the highest activation)
                        predicted_digit = numpy.argmax(y)

                        # Determine the actual digit (the one with 1 in the label vector)
                        actual_digit = numpy.argmax(label.transpose())

                        # Compare and count if the prediction is correct
                        if predicted_digit == actual_digit:
                            correct_predictions += 1

                    # Calculate accuracy based on the number of samples tested
                    accuracy = (correct_predictions / num_test_samples) * 100 if num_test_samples > 0 else 0.0
                    print(f"Accuracy on {num_test_samples} test samples: {accuracy:.2f}%")
                    print(f"Training Time: {training_time:.2f} seconds")


                    # Store the results
                    results.append({
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "hidden_layer_size": hidden_size,
                        "training_time": training_time,
                        "accuracy": accuracy
                    })

    # --- Save results to CSV ---
    csv_file_path = "nn_training_results.csv"
    if results:
        # Define the fieldnames for the CSV header
        fieldnames = ["epochs", "batch_size", "learning_rate", "hidden_layer_size", "training_time", "accuracy"]
        try:
            with open(csv_file_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader() # Write the header row
                writer.writerows(results) # Write all the data rows

            print(f"\nExperiment results saved to {csv_file_path}")
        except IOError as e:
            print(f"\nError saving results to CSV: {e}")
    else:
        print("\nNo results to save.")

=======
    
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
    
>>>>>>> 03c75fe8058c1a1b796dc93635380a9cc4155fda

if __name__ == '__main__':
    main()
