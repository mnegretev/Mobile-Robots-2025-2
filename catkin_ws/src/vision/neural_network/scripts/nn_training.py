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
import plotly.graph_objects as go
import os
import json

NAME = "Carlos Castañeda Mora"

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
        y = []
        #
        # TODO:
        # Write a function similar to 'forward' but instead of returning only the output layer,
        # return a list containing the output of each layer, from input to output.
        # Include input x as the first output.
        #
        
        y.append(x)
        current_output = x
        for i in range(len(self.biases)):
            u = numpy.dot(self.weights[i], current_output) + self.biases[i]
            current_output = 1.0 / (1.0 + numpy.exp(-u))
            y.append(current_output)
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

        delta = (y[-1] - t) * y[-1] * (1 - y[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, y[-2].transpose())
        
        for i in range(2, self.num_layers):
            delta = numpy.dot(self.weights[-i+1].transpose(), delta) * y[-i] * (1 - y[-i])
            nabla_b[-i] = delta
            nabla_w[-i] = numpy.dot(delta, y[-i-1].transpose())
        
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
def test_network(nn, testing_dataset, num_tests=100):
    correct = 0
    for i in range(num_tests):
        img, label = testing_dataset[numpy.random.randint(0, len(testing_dataset))]
        y = nn.forward(img)
        if numpy.argmax(y) == numpy.argmax(label):
            correct += 1
    return correct / num_tests * 100

def run_experiments(training_dataset, testing_dataset, architectures, learning_rates, epochs_list, batch_sizes, num_tests=100):
    results = []
    
    for architecture in architectures:
        for eta in learning_rates:
            for epochs in epochs_list:
                for batch_size in batch_sizes:
                    print(f"\nRunning experiment with architecture: {architecture}, eta: {eta}, epochs: {epochs}, batch_size: {batch_size}")
                    
                    # Train network
                    start_time = time.time()
                    nn = NeuralNetwork(architecture)
                    nn.train_by_SGD(training_dataset, epochs, batch_size, eta)
                    training_time = time.time() - start_time
                    
                    # Test network
                    success_rate = test_network(nn, testing_dataset, num_tests)
                    
                    # Store results as a dictionary
                    results.append({
                        'architecture': str(architecture),
                        'learning_rate': eta,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'training_time': training_time,
                        'success_rate': success_rate
                    })
                    
                    print(f"Results - Time: {training_time:.2f}s, Success: {success_rate:.2f}%")
    
    return results

def save_results(results, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    
    with open(os.path.join(folder, 'experiment_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    
    create_plots(results, folder)

def create_plots(results, folder):
    learning_rates = numpy.array([r['learning_rate'] for r in results])
    batch_sizes = numpy.array([r['batch_size'] for r in results])
    success_rates = numpy.array([r['success_rate'] for r in results])
    training_times = numpy.array([r['training_time'] for r in results])
    epochs = numpy.array([r['epochs'] for r in results])
    architectures = numpy.array([r['architecture'] for r in results])
    
    # Plot 1: Success rate vs learning rate for different batch sizes
    fig1 = go.Figure()
    unique_batch_sizes = numpy.unique(batch_sizes)
    
    for bs in unique_batch_sizes:
        mask = batch_sizes == bs
        fig1.add_trace(go.Scatter(
            x=learning_rates[mask],
            y=success_rates[mask],
            mode='lines+markers',
            name=f'Batch size {bs}',
            hovertemplate='Learning rate: %{x}<br>Success rate: %{y:.2f}%'
        ))
    
    fig1.update_layout(
        title='Success Rate vs Learning Rate for Different Batch Sizes',
        xaxis_title='Learning Rate',
        yaxis_title='Success Rate (%)',
        legend_title='Batch Size'
    )
    fig1.write_html(os.path.join(folder, 'success_vs_learning_rate.html'))
    
    # Plot 2: Training time vs epochs for different architectures
    fig2 = go.Figure()
    unique_architectures = numpy.unique(architectures)
    
    for arch in unique_architectures:
        mask = architectures == arch
        fig2.add_trace(go.Scatter(
            x=epochs[mask],
            y=training_times[mask],
            mode='lines+markers',
            name=f'Architecture {arch}',
            hovertemplate='Epochs: %{x}<br>Training time: %{y:.2f}s'
        ))
    
    fig2.update_layout(
        title='Training Time vs Epochs for Different Architectures',
        xaxis_title='Number of Epochs',
        yaxis_title='Training Time (s)',
        legend_title='Architecture'
    )
    fig2.write_html(os.path.join(folder, 'training_time_vs_epochs.html'))
    
    # Plot 3: Success rate heatmap (learning rate vs batch size)
    
    unique_lr = numpy.unique(learning_rates)
    unique_bs = numpy.unique(batch_sizes)
    heatmap_data = numpy.zeros((len(unique_lr), len(unique_bs)))
    
    for i, lr in enumerate(unique_lr):
        for j, bs in enumerate(unique_bs):
            mask = (learning_rates == lr) & (batch_sizes == bs)
            if numpy.any(mask):
                heatmap_data[i, j] = numpy.mean(success_rates[mask])
    
    fig3 = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=unique_bs,
        y=unique_lr,
        colorscale='Viridis',
        colorbar=dict(title='Success Rate (%)')
    ))
    fig3.update_layout(
        title='Success Rate Heatmap (Learning Rate vs Batch Size)',
        xaxis_title='Batch Size',
        yaxis_title='Learning Rate'
    )
    fig3.write_html(os.path.join(folder, 'success_rate_heatmap.html'))

def main():
    print("TRAINING A NEURAL NETWORK - " + NAME)
    rospy.init_node("nn_training")
    rospack = rospkg.RosPack()
    dataset_folder = rospack.get_path("neural_network") + "/handwritten_digits/"
    output_folder = rospack.get_path("neural_network") + "/experiment_results/"
    
    
    architectures = [
        [784, 30, 10],       
        [784, 100, 50, 10],  
        [784, 200, 10]       
    ]
    learning_rates = [0.5, 1.0, 3.0, 10.0]
    epochs_list = [3, 10, 50, 100]
    batch_sizes = [5, 10, 30, 100]
    num_tests = 100
    
    
    training_dataset, testing_dataset = load_dataset(dataset_folder)
    
    
    results = run_experiments(
        training_dataset, 
        testing_dataset, 
        architectures, 
        learning_rates, 
        epochs_list, 
        batch_sizes, 
        num_tests
    )
    
    
    save_results(results, output_folder)
    
    print("\nAll experiments completed. Results saved to:", output_folder)
    
    
    print("\nPress key to test a random network or ESC to exit...")
    numpy.set_printoptions(formatter={'float_kind':"{:.3f}".format})
    cmd = cv2.waitKey(0)
    while cmd != 27 and not rospy.is_shutdown():
        
        random_config = random.choice(results)
        nn = NeuralNetwork(eval(random_config['architecture']))
        nn.train_by_SGD(training_dataset, int(random_config['epochs']), 
                        int(random_config['batch_size']), random_config['learning_rate'])
        
        img, label = testing_dataset[numpy.random.randint(0, len(testing_dataset))]
        y = nn.forward(img).transpose()
        print(f"\nTesting network with config: {random_config}")
        print("Perceptron output: " + str(y))
        print("Expected output  : " + str(label.transpose()))
        print("Recognized digit : " + str(numpy.argmax(y)))
        cv2.imshow("Digit", numpy.reshape(numpy.asarray(img, dtype="float32"), (28,28,1)))
        cmd = cv2.waitKey(0)

if __name__ == '__main__':
    main()