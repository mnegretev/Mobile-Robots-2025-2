#!/usr/bin/env python
import rospy
import rospkg
import random
import numpy
import cv2
import torch

class MNIST(torch.nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.fc1 = torch.nn.Linear(784, 30)
        self.fc2 = torch.nn.Linear(30,10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.sigmoid(x)
        x = self.fc2(x)
        return torch.nn.functional.sigmoid(x)

def load_dataset(folder):
    print("Loading data set from " + folder)
    if not folder.endswith("/"):
        folder += "/"
    train_x, train_t, test_x, test_t = [],[],[],[]
    for i in range(10):
        f_data = [c/255.0 for c in open(folder + "data" + str(i), "rb").read(784000)]
        images = [numpy.asarray(f_data[784*j:784*(j+1)], dtype=numpy.float32).reshape([1,784]) for j in range(1000)]
        label  = numpy.asarray([1 if i == j else 0 for j in range(10)], dtype=numpy.float32).reshape([1,10])
        train_x += images[0:len(images)//2]
        train_t  += [label for j in range(len(images)//2)]
        test_x  += images[len(images)//2:len(images)]
        test_t   += [label for j in range(len(images)//2)]
    
    return numpy.asarray(train_x), numpy.asarray(train_t), numpy.asarray(test_x), numpy.asarray(test_t)

def train_by_SGD(model, train_x, train_t, epochs, batch_size, eta):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    for j in range(epochs):
        batches_x = [train_x[k:k+batch_size] for k in range(0,len(train_x), batch_size)]
        batches_t = [train_t[k:k+batch_size] for k in range(0,len(train_t), batch_size)]
        for batch_x, batch_t in zip(batches_x, batches_t):
            if rospy.is_shutdown():
                return
            optimizer.zero_grad()
            batch_y = model(torch.from_numpy(batch_x))
            loss = torch.nn.functional.mse_loss(batch_y, torch.from_numpy(batch_t))
            loss.backward()
            #sys.stdout.write("\rGradient magnitude: %f            " % (self.get_gradient_mag(nabla_w, nabla_b)))
            #sys.stdout.flush()
        print("Epoch: " + str(j))

def main():
    print("TRAINING A NEURAL NETWORK USING PYTORCH")
    rospy.init_node("nn_training")
    rospack = rospkg.RosPack()

    epochs        = 3
    batch_size    = 100
    learning_rate = 3.0
    if rospy.has_param("~epochs"):
        epochs = rospy.get_param("~epochs")
    if rospy.has_param("~batch_size"):
        batch_size = rospy.get_param("~batch_size")
    if rospy.has_param("~learning_rate"):
        learning_rate = rospy.get_param("~learning_rate") 
        
    dataset_folder = rospack.get_path("neural_network") + "/handwritten_digits/"
    nn = MNIST()
    train_x, train_t, test_x, test_t = load_dataset(dataset_folder)
    train_by_SGD(nn, train_x, train_t, epochs, batch_size, learning_rate)

    print("\nPress key to test network or ESC to exit...")
    numpy.set_printoptions(formatter={'float_kind':"{:.3f}".format})
    cmd = cv2.waitKey(0)
    while cmd != 27 and not rospy.is_shutdown():
        idx = numpy.random.randint(0, 4999)
        img = test_x[idx]
        t = test_t[idx]
        y = nn.forward(torch.from_numpy(img))
        print("\nPerceptron output: " + str(y))
        print("Expected output  : "   + str(t))
        print("Recognized digit : "   + str(numpy.argmax(torch.Tensor.numpy(y, force=True))))
        cv2.imshow("Digit", numpy.reshape(numpy.asarray(img, dtype="float32"), (28,28,1)))
        cmd = cv2.waitKey(0)


if __name__ == "__main__":
    main()
