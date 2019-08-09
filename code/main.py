import pickle
import numpy as np
from matplotlib import pyplot as plt
import csv
import tensorflow as tf
from LeNet_5 import LeNet
import random
from sklearn.utils import shuffle

# Load the Image data
training_file = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'


with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_train = (X_train.astype(float))/float(255)  # Normalize the data between 0-1
X_valid, y_valid = valid['features'], valid['labels']
X_valid = (X_valid.astype(float))/float(255)  # Normalize the data between 0-1
X_test, y_test = test['features'], test['labels']

# Load the cvs file into a dict
csv_file = './signnames.csv'
sign_lable_names = {}
line_no = 0
with open(csv_file, mode='r') as f:
    csv_data = csv.reader(f, delimiter=',')
    for line in csv_data:
        if line_no == 0:
            line_no = 1
            continue
        else:
            sign_lable_names[int(line[0])] = line[1]



# Get the different sizes
# Number of training examples
n_train = y_train.shape[0]

# Number of validation examples
n_validation = y_valid.shape[0]

# Number of testing examples.
n_test = y_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = str(X_test[0].shape[1]) + 'x' + str(X_test[0].shape[0])

# How many unique classes/labels there are in the dataset.
n_classes = len(sign_lable_names.keys())

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("Image Shape = ", X_test[0].shape)
print("Input Shape =", X_train.shape)
print("Output Shape =", y_train.shape)

# Hyperparameters to tune
EPOCHS = 50
BATCH_SIZE = 8
RATE = 0.0001
DROP_RATE = 0.5

# Create Placeholders
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
drop_rate = tf.placeholder(tf.float32, ())
one_hot_y = tf.one_hot(y, n_classes)

# Training pipeline
logits = LeNet(x, drop_rate)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=RATE)
training_operation = optimizer.minimize(loss_operation)

# # Loss function using L2 Regularization
# regularizer = tf.nn.l2_loss(weights)
# loss = tf.reduce_mean(loss + beta * regularizer)


# Evaluation the model
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, drop_rate:0.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, drop_rate:DROP_RATE})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        training_accuracy = evaluate(X_train, y_train)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

# Evaluate the model with the test images
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
    validation_accuracy = evaluate(X_valid, y_valid)
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))