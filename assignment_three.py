# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import math
from six.moves import cPickle as pickle

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 128
hidden_nodes_layer1 = 1024
hidden_nodes_layer2 = 512
hidden_nodes_layer3 = 512

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  lmda = tf.placeholder(tf.float32)
  keep_prob1 = tf.placeholder(tf.float32)
  keep_prob2 = tf.placeholder(tf.float32)
  #global_step = tf.Variable(0)  # count the number of steps taken.
  
  # Variables.
  stddev1 = math.sqrt(2/(image_size * image_size))
  stddev2 = math.sqrt(2/(hidden_nodes_layer1))
  stddev3 = math.sqrt(2/(hidden_nodes_layer2))
  stddev4 = math.sqrt(2/(hidden_nodes_layer3))
  weights_1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_nodes_layer1], stddev=stddev1))
  biases_1 = tf.Variable(tf.constant(0.1, shape=[hidden_nodes_layer1]))
  weights_2 = tf.Variable(
    tf.truncated_normal([hidden_nodes_layer1, hidden_nodes_layer2], stddev=stddev2))
  biases_2 = tf.Variable(tf.constant(0.1, shape=[hidden_nodes_layer2]))
  weights_3 = tf.Variable(
    tf.truncated_normal([hidden_nodes_layer2, hidden_nodes_layer3], stddev=stddev3))
  biases_3 = tf.Variable(tf.constant(0.1, shape=[hidden_nodes_layer3]))
  weights_4 = tf.Variable(
    tf.truncated_normal([hidden_nodes_layer3, num_labels], stddev=stddev4))
  biases_4 = tf.Variable(tf.constant(0.1, shape=[num_labels]))
  
  # Training computation.
  def logits(dataset):
  	hypoth_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(dataset, weights_1) + biases_1), keep_prob1)
  	hypoth_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(hypoth_1, weights_2) + biases_2), keep_prob2)
  	hypoth_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(hypoth_2, weights_3) + biases_3), keep_prob2)
  	return tf.matmul(hypoth_3, weights_4) + biases_4
  #hypoth_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
  #logits = tf.matmul(hypoth_1, weights_2) + biases_2
  #logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits(tf_train_dataset), tf_train_labels)) + lmda * (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) +
    tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4))
  
  # Optimizer.
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  #optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  optimizer = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(loss)
  #optimizer = tf.train.AdagradOptimizer(learning_rate=0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits(tf_train_dataset))
  valid_prediction = tf.nn.softmax(logits(tf_valid_dataset))
  test_prediction = tf.nn.softmax(logits(tf_test_dataset))
  

num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, lmda : 0, keep_prob1 : 1, keep_prob2 : 1}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(feed_dict = {keep_prob1 : 1, keep_prob2 : 1}), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(feed_dict = {keep_prob1 : 1, keep_prob2 : 1}), test_labels))
