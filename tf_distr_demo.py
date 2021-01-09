# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import device_lib

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

learning_rate = 0.001
training_steps = 8250
batch_size = 100
display_step = 100

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def build_model():

    def multilayer_perceptron(x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)

        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    with tf.variable_scope('aaa'):
        weights = {
        'h1': _variable_on_cpu('h1',[n_input, n_hidden_1],tf.random_normal_initializer()),
        'h2': _variable_on_cpu('h2',[n_hidden_1, n_hidden_2],tf.random_normal_initializer()),
        'out': _variable_on_cpu('out_w',[n_hidden_2, n_classes],tf.random_normal_initializer())
          }
        biases = {
        'b1': _variable_on_cpu('b1',[n_hidden_1],tf.random_normal_initializer()),
        'b2': _variable_on_cpu('b2',[n_hidden_2],tf.random_normal_initializer()),
        'out': _variable_on_cpu('out_b',[n_classes],tf.random_normal_initializer())
          }

        pred = multilayer_perceptron(x, weights, biases)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    return cost,pred


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g,_ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


with tf.Graph().as_default(), tf.device('/cpu:0'):
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    tower_grads = []
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(num_gpus):
        with tf.device('/gpu:%d' % i):
                cost,pred = build_model()
                tf.get_variable_scope().reuse_variables()
                grads = optimizer.compute_gradients(cost)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = optimizer.apply_gradients(grads)
    train_op = apply_gradient_op

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for step in range(training_steps):
            image_batch, label_batch = mnist.train.next_batch(batch_size)
            _, cost_print = sess.run([train_op, cost],
                                     {x:image_batch,
                                      y:label_batch})

            if step % display_step == 0:
                print("step=%04d" % (step+1)+  " cost=" + str(cost_print))
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    with sess.as_default():
      print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    sess.close()