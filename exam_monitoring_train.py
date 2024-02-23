"""
Group Project (CSC746)
Project Title: Exam Proctoring System Using Face Detection
by Rajini Chittimalla, Sujung Choi, Madhu Sai Vineel Reka
File Description: This file is to build the model to predict the facial landmarks using the trained model.
"""
import numpy as np
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
import random

# Set random seed for NumPy
np.random.seed(42)

# Set random seed for Python's random module
random.seed(42)

# Set random seed for TensorFlow
tf.random.set_seed(42)

ACCURACY_THRESHOLD = 0.9998 # set threshold to stop training when accuracy 99.98%
class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        """
        # on_epoch_end() function is used to stop the training when the accuracy reaches the threshold
        #
        # Input
        ----------------
        # epoch: number of epochs
        # logs: logs for the training
        """
        if(logs.get('accuracy') >= ACCURACY_THRESHOLD):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD * 100))
            self.model.stop_training = True

callbacks = myCallback()


def load_prepared_data(train_filename, test_filename, l):
    """
    # load_prepared_data() function is used to load the prepared data
    #
    # Input
    ----------------
    # train_filename: training data file name
    # test_filename: testing data file name
    # l: label
    #
    # Output
    ----------------
    # train: training data
    # tr_labels: training labels
    # test: testing data
    # ts_labels: testing labels
    """
    train = np.loadtxt(train_filename, delimiter=',', converters=float, skiprows=1)
    test  = np.loadtxt(test_filename, delimiter=',', converters=float, skiprows=1)
    train = train[:,1:]
    test = test[:,1:]
    
    tr_labels = np.zeros(train.shape[0]) + l 
    ts_labels = np.zeros(test.shape[0]) + l 

    return train, tr_labels, test, ts_labels

def generate_model(input_sz, num_classes, learning_rate):
    """
    # generate_model() function is used to generate the neural network model
    #
    # Input
    #----------------
    # input_sz: input size
    # num_classes: number of classes
    # learning_rate: learning rate for optimization
    #
    # Output
    #----------------
    # nn_model: neural network model
    """
    inputs = Input(shape=input_sz)  
    
    L1  = Dense(400, activation = 'relu')(inputs)
    L2  = Dense(200, activation = 'relu')(L1)
    L3  = Dense(100, activation = 'relu')(L2)
    L4  = Dense(50, activation = 'relu')(L3)
    L5  = Dense(32, activation = 'relu')(L4)
    L6  = Dense(16, activation = 'relu')(L5)
    
    L7  = Dense(num_classes, activation='softmax')(L6)
        
    nn_model = Model(inputs=inputs, outputs=L7)
    
    # compile the model and calculate the accuracy
    nn_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
    nn_model.summary() # print the model summary
        
    return nn_model

def save_models(learned_model):
    """
    # save_models() function is used to save the trained model.
    #
    # Input
    ----------------
    # learned_model: trained model
    """
    filename = "models/class_monitor_3class_1.h5"
    learned_model.save(filename)   

    
def main():
    """
    # main() function is to call the functions to 1) load the prepared data, 2) generate the model, 
    # 3) train and test the model, and 4) display the classification error and accuracy.
    """

    # load the prepared data
    n_tr_data, n_tr_labels, n_ts_data, n_ts_labels = load_prepared_data('dataset/normal.csv', 'dataset/normal_test.csv',0)
    r_tr_data, r_tr_labels, r_ts_data, r_ts_labels = load_prepared_data('dataset/right.csv', 'dataset/right_test.csv',1)
    l_tr_data, l_tr_labels, l_ts_data, l_ts_labels = load_prepared_data('dataset/left.csv', 'dataset/left_test.csv',2)

    # combine the data and labels for training and testing
    All_tr_data = np.vstack((n_tr_data, r_tr_data,l_tr_data))
    All_ts_data = np.vstack((n_ts_data, r_ts_data,l_ts_data))

    All_tr_labels = np.hstack((n_tr_labels, r_tr_labels, l_tr_labels))
    All_ts_labels = np.hstack((n_ts_labels, r_ts_labels, l_ts_labels))

    # set the number of classes to 3, including normal, right, and left
    num_classes = 3

    # set the input size to 12, which is the number of facial landmarks
    input_size = 12

    tr_labels = to_categorical(All_tr_labels)
    ts_labels = to_categorical(All_ts_labels)

    # set the hyperparameters
    batch_size = 32
    epochs = 350
    learning_rate=0.0005

    #train
    tr_features = All_tr_data.reshape(-1, input_size, 1)

    #test
    ts_features = All_ts_data.reshape(-1, input_size, 1)

    nn_model = generate_model(input_size, num_classes, learning_rate)
    nn_model.fit(tr_features, tr_labels, batch_size=batch_size, epochs=epochs, callbacks=[callbacks], verbose=1)

    test_eval = nn_model.evaluate(ts_features, ts_labels, verbose=0)
    print("Classification error:", test_eval[0])
    print("Classification accuracy:", test_eval[1] * 100)

    # save the trained model to the local directory
    save_models(nn_model)

main()