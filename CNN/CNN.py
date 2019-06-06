import cv2
import os
import numpy as np
import tflearn
from tqdm import tqdm
from random import shuffle
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class CNN:
    LR = 1e-3
    IMG_SIZE = 150
    TRAIN_DIR = ''
    TEST_DIR = 'E:\\4th year\\1st term\\Project\\prototype\\CNN\\dataset_canny\\test_set\\test'
    MODEL_NAME = 'handRecognitionCanny2.model'.format(LR, '5conv-basic')
    convolution_network = ""

    def __init__(self):
        convnet = input_data(shape=[None, self.IMG_SIZE, self.IMG_SIZE, 1], name='input')

        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = fully_connected(convnet, 512, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 26, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=self.LR,
                             loss='categorical_crossentropy', name='targets')

        self.convolution_network = convnet

    def label_img(self, img):
        word_label = img.split('.')[0]
        # DIY One hot encoder
        print(word_label)
        if word_label == 'A':
            return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'B':
            return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'C':
            return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'D':
            return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'E':
            return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'F':
            return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'G':
            return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'H':
            return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'I':
            return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'J':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'K':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'L':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'M':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'N':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'O':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'P':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'Q':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'R':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'S':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif word_label == 'T':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif word_label == 'U':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif word_label == 'V':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif word_label == 'W':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif word_label == 'X':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif word_label == 'Y':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif word_label == 'Z':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    def process_train_data(self):
        training_data = []

        for img in tqdm(os.listdir(self.TRAIN_DIR)):
            label = self.label_img(img)
            path = os.path.join(self.TRAIN_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
        np.random.shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data

    def process_test_data(self):
        testing_data = []
        for img in tqdm(os.listdir(self.TEST_DIR)):
            label = self.label_img(img)
            path = os.path.join(self.TEST_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            testing_data.append([np.array(img), label])

        shuffle(testing_data)
        np.random.shuffle(testing_data)
        np.save('test_data.npy', testing_data)
        return testing_data

    def accuracy(self, test_data, model):
        accuracy_value = 0
        for data in test_data:
            img_data = data[0]
            label = data[1]
            data = img_data.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)
            model_out = model.predict([data])[0]
            if np.argmax(model_out) == np.argmax(label):
                accuracy_value = accuracy_value + 1
        accuracy_value = float(accuracy_value) / len(test_data)
        print("accuracy is ", accuracy_value)

    def convolution_neural_network(self):
        train_data = self.process_train_data()
        # train_data = np.load('train_data.npy')

        model = tflearn.DNN(self.convolution_network, tensorboard_dir='log')

        train = train_data[:-500]
        test = train_data[-500:]

        X = np.array([i[0] for i in train]).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        Y = [i[1] for i in train]
        test_x = np.array([i[0] for i in test]).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=5,
                  validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=500, show_metric=True, run_id=self.MODEL_NAME)
        model.save(self.MODEL_NAME)

    def test_convolution_neural_network(self, img):
        model = tflearn.DNN(self.convolution_network, tensorboard_dir='log')
        model.load(self.MODEL_NAME)

        data = img.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)
        # tf.reset_default_graph()
        model_out = model.predict([data])[0]

        str_label = ""
        if np.argmax(model_out) == 0:
            str_label = "A"
        elif np.argmax(model_out) == 1:
            str_label = "B"
        elif np.argmax(model_out) == 2:
            str_label = "C"
        elif np.argmax(model_out) == 3:
            str_label = "D"
        elif np.argmax(model_out) == 4:
            str_label = "E"
        elif np.argmax(model_out) == 5:
            str_label = "F"
        elif np.argmax(model_out) == 6:
            str_label = "G"
        elif np.argmax(model_out) == 7:
            str_label = "H"
        elif np.argmax(model_out) == 8:
            str_label = "I"
        elif np.argmax(model_out) == 9:
            str_label = "J"
        elif np.argmax(model_out) == 10:
            str_label = "K"
        elif np.argmax(model_out) == 11:
            str_label = "L"
        elif np.argmax(model_out) == 12:
            str_label = "M"
        elif np.argmax(model_out) == 13:
            str_label = "N"
        elif np.argmax(model_out) == 14:
            str_label = "O"
        elif np.argmax(model_out) == 15:
            str_label = "P"
        elif np.argmax(model_out) == 16:
            str_label = "Q"
        elif np.argmax(model_out) == 17:
            str_label = "R"
        elif np.argmax(model_out) == 18:
            str_label = "S"
        elif np.argmax(model_out) == 19:
            str_label = "T"
        elif np.argmax(model_out) == 20:
            str_label = "U"
        elif np.argmax(model_out) == 21:
            str_label = "V"
        elif np.argmax(model_out) == 22:
            str_label = "W"
        elif np.argmax(model_out) == 23:
            str_label = "X"
        elif np.argmax(model_out) == 24:
            str_label = "Y"
        elif np.argmax(model_out) == 25:
            str_label = "Z"
        return str_label

    def test_convolution_neural_network_accuracy(self):
        model = tflearn.DNN(self.convolution_network, tensorboard_dir='log')
        model.load(self.MODEL_NAME)

        test_data = self.process_test_data()

        self.accuracy(test_data, model)

# cnn = CNN()
# cnn.test_convolution_neural_network_accuracy()
