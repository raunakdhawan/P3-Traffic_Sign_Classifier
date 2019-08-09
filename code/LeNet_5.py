'''
Architecture
- Layer 1 : Convolutional. The output shape should be 28x28x6.
- Activation : Your choice of activation function.
- Pooling : The output shape should be 14x14x6.
- Layer 2 : Convolutional. The output shape should be 10x10x16.
- Activation : Your choice of activation function.
- Pooling : The output shape should be 5x5x16.
- Flatten : Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. 
The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.
- Layer 3 : Fully Connected. This should have 120 outputs.
- Activation : Your choice of activation function.
- Layer 4 : Fully Connected. This should have 84 outputs.
- Activation : Your choice of activation function.
- Layer 5 : Fully Connected (Logits). This should have 10 outputs.
'''
import tensorflow as tf



def LeNet(x):
    # For normalization of the input features
    mu = 0
    sigma = 0.1

    # Convolution 1 + Activation + Pooling
    conv_1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    conv_1_b = tf.Variable(tf.zeros(6))
    conv_1 = tf.nn.conv2d(x, conv_1_w, strides=[1, 1, 1, 1], padding='VALID') + conv_1_b  # Conv
    conv_1 = tf.nn.relu(conv_1)  # Activation
    conv_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # Pooling

    # Convolution 2 + Activation + Pooling
    conv_2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv_2_b = tf.Variable(tf.zeros(16))
    conv_2 = tf.nn.conv2d(conv_1, conv_2_w, strides=[1, 1, 1, 1], padding='VALID') + conv_2_b  # Conv
    conv_2 = tf.nn.relu(conv_2)  # Activation
    conv_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # Pooling

    # Flatten the last conv layer
    fc_0 = tf.contrib.layers.flatten(conv_2)

    # Layer 3
    fc3_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(120))
    fc3   = tf.matmul(fc_0, fc3_W) + fc3_b
    fc3 = tf.nn.relu(fc3)

    # Layer 4
    fc4_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc4_b  = tf.Variable(tf.zeros(84))
    fc4    = tf.matmul(fc3, fc4_W) + fc4_b
    fc4    = tf.nn.relu(fc4)

    # Layer 5
    fc5_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc5_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc4, fc5_W) + fc5_b

    return logits


