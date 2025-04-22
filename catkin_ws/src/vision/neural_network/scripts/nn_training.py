#!/usr/bin/env python3

import cv2
import sys
import time
import random
import numpy
import rospy
import rospkg

NAME = "MURILLO SANTOS JAVIER EDUARDO"

class NeuralNetwork(object):
    def __init__(self, layers, weights=None, biases=None):
        self.num_layers  = len(layers)
        self.layer_sizes = layers
        self.biases = [numpy.random.randn(y, 1) for y in layers[1:]] if biases is None else biases
        self.weights = [numpy.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])] if weights is None else weights

    def feedforward(self, x):
        for i in range(len(self.biases)):
            z = numpy.dot(self.weights[i], x) + self.biases[i]
            x = 1.0 / (1.0 + numpy.exp(-z))
        return x

    def feedforward_verbose(self, x):
        y = [x]
        for i in range(len(self.biases)):
            z = numpy.dot(self.weights[i], x) + self.biases[i]
            x = 1.0 / (1.0 + numpy.exp(-z))
            y.append(x)
        return y

    def backpropagate(self, x, yt):
        y = self.feedforward_verbose(x)
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        delta = (y[-1] - yt) * y[-1] * (1 - y[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, y[-2].T)

        for l in range(2, self.num_layers):
            z = y[-l]
            sp = z * (1 - z)
            delta = numpy.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, y[-l-1].T)

        return nabla_w, nabla_b

    def update_with_batch(self, batch, eta):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        M = len(batch)
        for x, y in batch:
            if rospy.is_shutdown():
                break
            delta_nabla_w, delta_nabla_b = self.backpropagate(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - eta * nw / M for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - eta * nb / M for b, nb in zip(self.biases , nabla_b)]
        return nabla_w, nabla_b

    def get_gradient_mag(self, nabla_w, nabla_b):
        mag_w = sum([numpy.sum(n) for n in [nw * nw for nw in nabla_w]])
        mag_b = sum([numpy.sum(b) for b in [nb * nb for nb in nabla_b]])
        return mag_w + mag_b

    def train_by_SGD(self, training_data, epochs, batch_size, eta):
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
            for batch in batches:
                if rospy.is_shutdown():
                    return
                nabla_w, nabla_b = self.update_with_batch(batch, eta)
                sys.stdout.write("\rGradient magnitude: %f            " % (self.get_gradient_mag(nabla_w, nabla_b)))
                sys.stdout.flush()
            print("Epoch: " + str(j))


def load_dataset(folder):
    print("Loading data set from " + folder)
    if not folder.endswith("/"):
        folder += "/"
    training_dataset, training_labels, testing_dataset, testing_labels = [], [], [], []
    for i in range(10):
        f_data = [c / 255.0 for c in open(folder + "data" + str(i), "rb").read(784000)]
        images = [numpy.asarray(f_data[784 * j:784 * (j + 1)]).reshape([784, 1]) for j in range(1000)]
        label  = numpy.asarray([1 if i == j else 0 for j in range(10)]).reshape([10, 1])
        training_dataset += images[0:len(images) // 2]
        training_labels  += [label for j in range(len(images) // 2)]
        testing_dataset  += images[len(images) // 2:len(images)]
        testing_labels   += [label for j in range(len(images) // 2)]
    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))


def evaluar_red(nn, testing_dataset, cantidad=100):
    correctos = 0
    errores = 0

    print(f"\nEvaluando {cantidad} imágenes de prueba...\n")
    for _ in range(cantidad):
        img, label = random.choice(testing_dataset)
        y = nn.feedforward(img).transpose()
        expected = numpy.argmax(label)
        predicted = numpy.argmax(y)
        print(f"Perceptron output: {y}")
        print(f"Expected output  : {label.transpose()}")
        print(f"Recognized digit : {predicted}")
        if expected == predicted:
            print("✔️ Correcto\n")
            correctos += 1
        else:
            print("❌ Incorrecto\n")
            errores += 1

    print("="*50)
    print(f"✅ Casos correctos: {correctos}")
    print(f"❌ Casos incorrectos: {errores}")
    print(f"📊 Precisión total: {correctos / cantidad * 100:.2f}%")
    print("="*50)
    return correctos, errores


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

    try:
        saved_data = numpy.load(dataset_folder + "network.npz", allow_pickle=True)
        layers = [saved_data['w'][0].shape[1]] + [b.shape[0] for b in saved_data['b']]
        nn = NeuralNetwork(layers, weights=saved_data['w'], biases=saved_data['b'])
        print("Loading data from previously trained model with layers " + str(layers))
    except:
        nn = NeuralNetwork([784, 30, 10])
        pass

    start_time = time.time()
    nn.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)
    end_time = time.time()
    print(f"\nTiempo total de entrenamiento: {end_time - start_time:.2f} segundos")

    evaluar_red(nn, testing_dataset, cantidad=100)


if __name__ == '__main__':
    main()
