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
import os

NAME = "Camarena Olivos Alan Misael"

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
       #asdasd 
    def forward(self, x):
        #
        # This function gets the output of the network when input is 'x'.
        #
        for i in range(len(self.biases)):
            u = numpy.dot(self.weights[i], x) + self.biases[i]
            x = 1.0 / (1.0 + numpy.exp(-u))  #output of the current layer is the input of the next one
        return x

    def feedforward_verbose(self, x):
        y = [x]
        #
        # TODO:
        # Write a function similar to 'forward' but instead of returning only the output layer,
        # return a list containing the output of each layer, from input to output.
        # Include input x as the first output.
        #for i in range(len(self.biases)):
        for i in range(len(self.biases)):
            z = numpy.dot(self.weights[i], y[-1]) + self.biases[i]
            y.append(1.0 / (1.0 + numpy.exp(-z)))  
        return y

    def backpropagate(self, x, t):
        y = self.feedforward_verbose(x)
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

        for l in range(2, self.num_layers):
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * y[-l] * (1 - y[-l])
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
                sys.stdout.write("\rGradient magnitude: %f            " % (self.get_gradient_mag(nabla_w, nabla_b)))
                sys.stdout.flush()
            print("Epoch: " + str(j))
    #
    ### END OF CLASS
    #

def evaluate_performance(nn, testing_dataset, num_tests=100):
    correct = 0
    for _ in range(num_tests):
        img, label = testing_dataset[numpy.random.randint(0, 4999)]
        y = nn.forward(img)
        if numpy.argmax(y) == numpy.argmax(label):
            correct += 1
    return (correct / num_tests) * 100

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
    
   
    architecture = [784, 30, 10]
    
    # parametros
    learning_rates = [0.5, 1.0, 3.0, 10.0]
    epochs_list = [3, 10, 50, 100]
    batch_sizes = [5, 10, 30, 100]
    num_tests = 100  #pruebas para cada configuración
    
    training_dataset, testing_dataset = load_dataset(dataset_folder)
    
    
    results_file = dataset_folder + "results/nn_results.txt"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("RESULTADOS DE EXPERIMENTOS RED NEURONAL\n")
        f.write("=" * 50 + "\n")

    for lr in learning_rates:
        for eps in epochs_list:
            for bs in batch_sizes:
                print(f"\nPrueba con configuración:")
                print(f"Tasa de aprendizaje: {lr}")
                print(f"Épocas: {eps}")
                print(f"Tamaño de lote: {bs}")
                
                total_success_rate = 0
                total_training_time = 0
                
                # for para realizar 100 pruebas en cad una
                for test in range(num_tests):
                    print(f"Prueba {test + 1}/{num_tests}")
                    
                    # entrenamiento 
                    start_time = time.time()
                    nn = NeuralNetwork(architecture)
                    nn.train_by_SGD(training_dataset, eps, bs, lr)
                    training_time = time.time() - start_time
                    
                    
                    success_rate = evaluate_performance(nn, testing_dataset)
                    
                    total_success_rate += success_rate
                    total_training_time += training_time
                
                # calcular promedios
                avg_success_rate = total_success_rate / num_tests
                avg_training_time = total_training_time / num_tests
                
                
                with open(results_file, 'a') as f:
                    f.write(f"\nConfiguración:\n")
                    f.write(f"Tasa de aprendizaje: {lr}\n")
                    f.write(f"Épocas: {eps}\n")
                    f.write(f"Tamaño de lote: {bs}\n")
                    f.write(f"Tiempo promedio de entrenamiento: {avg_training_time:.2f} segundos\n")
                    f.write(f"Tasa promedio de éxito: {avg_success_rate:.2f}%\n")
                    f.write("-" * 50 + "\n")
                
                if rospy.is_shutdown():
                    return
    
    print(f"\nExperimentos completados. Resultados guardados en: {results_file}")  
#prueba para ver que sucede error github
#cambio por que no se que sucede
#no se que sucede
if __name__ == '__main__':
    main()
