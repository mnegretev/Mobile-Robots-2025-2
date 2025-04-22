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
import time
import csv
import rospy
import rospkg

NAME = "Frausto Martinez Juan Carlos"

class NeuralNetwork(object):
        #
        # The list 'layers' indicates the number of neurons in each layer.
        # Remember that the first layer indicates the dimension of the inputs and thus,
        # there is no bias vector fot the first layer.
        # For this practice, 'layers' should be something like [784, n2, n3, ..., nl, 10]
        # All weights and biases are initialized with random values. In each layer we have a matrix
        # of weights where row j contains all the weights of the j-th neuron in that layer. For this example,
        # the first matrix should be of order n2 x 784 and last matrix should be 10 x nl.
        #
    def __init__(self, layers, weights=None, biases=None):
        self.num_layers  = len(layers)
        self.layer_sizes = layers
        self.biases  = [numpy.random.randn(y, 1) for y in layers[1:]] if biases is None else biases
        self.weights = [numpy.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])] if weights is None else weights


    def forward(self, x):
        #
        # This function gets the output of the network when input is 'x'.
        #
        for b, w in zip(self.biases, self.weights):
            x = 1.0 / (1.0 + numpy.exp(-(numpy.dot(w, x) + b)))
        return x

    def forward_all_outputs(self, x):
        # Write a function similar to 'forward' but instead of returning only the output layer,
        # return a list containing the output of each layer, from input to output.
        # Include input x as the first output.
        #
        y = [x]
        for b, w in zip(self.biases, self.weights):
            x = 1.0 / (1.0 + numpy.exp(-(numpy.dot(w, x) + b)))
            y.append(x)
        return y

    def backpropagate(self, x, t):
        y = self.forward_all_outputs(x)
        nabla_b = [numpy.zeros_like(b) for b in self.biases]
        nabla_w = [numpy.zeros_like(w) for w in self.weights]
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
        nabla_w[-1] = numpy.dot(delta, y[-2].T)

        for l in range(2, self.num_layers):
            z = y[-l]
            delta = numpy.dot(self.weights[-l + 1].T, delta) * z * (1 - z)
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, y[-l - 1].T)
        return nabla_w, nabla_b

    def update_with_batch(self, batch, eta):
        #
        # This function exectutes gradient descend for the subset of examples
        # given by 'batch' with learning rate 'eta'
        # 'batch' is a list of training examples [(x,y), ..., (x,y)]
        #
        nabla_b = [numpy.zeros_like(b) for b in self.biases]
        nabla_w = [numpy.zeros_like(w) for w in self.weights]
        m = len(batch)
        for x, y in batch:
            if rospy.is_shutdown():
                break
            delta_w, delta_b = self.backpropagate(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_b)]
        self.weights = [w - eta * nw / m for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - eta * nb / m for b, nb in zip(self.biases, nabla_b)]
        return nabla_w, nabla_b

    def get_gradient_mag(self, nabla_w, nabla_b):
        mag_w = sum(numpy.sum(nw * nw) for nw in nabla_w)
        mag_b = sum(numpy.sum(nb * nb) for nb in nabla_b)
        return mag_w + mag_b

    def train_by_SGD(self, training_data, epochs, batch_size, eta):
        grad_mag = 0.0
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
            for batch in batches:
                if rospy.is_shutdown():
                    return 0.0
                nabla_w, nabla_b = self.update_with_batch(batch, eta)
                grad_mag = self.get_gradient_mag(nabla_w, nabla_b)  # guarda el gradiente más reciente
                sys.stdout.write("\rGradient magnitude: %f            " % grad_mag)
                sys.stdout.flush()
            print("Epoch:", j)
        return grad_mag  # magnitud final

    #
    ### END OF CLASS
    #

    
def load_dataset(folder):
    if not folder.endswith("/"):
        folder += "/"
    training_dataset, training_labels, testing_dataset, testing_labels = [], [], [], []
    for i in range(10):
        f_data = [c / 255.0 for c in open(folder + f"data{i}", "rb").read(784000)]
        images = [numpy.asarray(f_data[784 * j:784 * (j + 1)]).reshape([784, 1]) for j in range(1000)]
        label = numpy.asarray([1 if i == j else 0 for j in range(10)]).reshape([10, 1])
        half = len(images) // 2
        training_dataset += images[:half]
        training_labels  += [label] * half
        testing_dataset  += images[half:]
        testing_labels   += [label] * half
    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))

def run_experiment(nn_arch, train_data, test_data, epochs, batch_size, lr):
    nn = NeuralNetwork(nn_arch)

    t0 = time.time()
    final_grad_mag = nn.train_by_SGD(train_data, epochs, batch_size, lr)
    training_time = time.time() - t0

    t_eval0 = time.time()
    correct = 0
    for _ in range(100):
        img, lbl = random.choice(test_data)
        y = nn.forward(img)
        if numpy.argmax(y) == numpy.argmax(lbl):
            correct += 1
    eval_time = time.time() - t_eval0

    return training_time, eval_time, final_grad_mag, correct


def main():
    print("TRAINING A NEURAL NETWORK –", NAME)
    rospy.init_node("nn_training_auto", anonymous=True)

    rospack = rospkg.RosPack()
    dataset_folder = rospack.get_path("neural_network") + "/handwritten_digits/"
    train_ds, test_ds = load_dataset(dataset_folder)

    learning_rates = [0.5, 1.0, 3.0, 10.0]
    epochs_list    = [3, 10, 50, 100]
    batch_sizes    = [5, 10, 30, 100]

    nn_arch = [784, 64, 32, 10]

    csv_file = "experiment_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "learning_rate", "epochs", "batch_size",
            "training_time_s", "eval_time_s", "grad_mag_final", "successes_out_of_100"
        ])

        total = len(learning_rates) * len(epochs_list) * len(batch_sizes)
        idx = 1
        for lr in learning_rates:
            for ep in epochs_list:
                for bs in batch_sizes:
                    if rospy.is_shutdown():
                        print("\nROS apagado – abortando…")
                        return
                    print(f"\nCombo {idx}/{total}: lr={lr}, epochs={ep}, batch={bs}")
                    idx += 1
                    t_train, t_eval, gmag, hits = run_experiment(nn_arch, train_ds, test_ds, ep, bs, lr)
                    writer.writerow([lr, ep, bs, round(t_train, 3), round(t_eval, 3), round(gmag, 6), hits])
                    f.flush()
    print(f"\nExperimentos terminados. CSV en '{csv_file}'.")

if __name__ == "__main__":
    main()
