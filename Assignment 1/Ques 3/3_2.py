import tensorflow as tf
import visual as vis
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np

X_train = np.load('Xtrain.npy')
Y_train = np.load('Ytrain.npy')
X_test = np.load('Xtest.npy')
Y_test = np.load('Ytest.npy')

weights = list()
biases = list()

def FC(input1, nn, activation, use_bias, islast):
    dim_one = input1.get_shape().as_list()
    dim_two = nn
    w = tf.Variable(tf.truncated_normal([dim_one[1], dim_two], stddev=0.1))
    b = tf.Variable(tf.zeros([dim_two]))
    if (use_bias):
        y = tf.matmul(input1, w) + b
    else:
        y = tf.matmul(input1, w)
    if (islast):
        return y
    else:
        if activation == 'sigmoid':
            return tf.nn.sigmoid(y)
        elif activation == 'relu':
            return tf.nn.relu(y)    
    
epochs = 5
step = 100
batch = 100

# mnist = read_data_sets("MNIST", one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 3])
Y_ = tf.placeholder(tf.float32, [None, 96]) # correct labels

XX = tf.reshape(X, [-1, 2352])

Y = FC(XX, 400, 'sigmoid', True, False)
Y = FC(Y, 300, 'sigmoid', True, False)
Y = FC(Y, 150, 'sigmoid', True, False)
Y = FC(Y, 100, 'sigmoid', True, False)
Y_dash = FC(Y, 96, 'sigmoid', True, True)
Y = tf.nn.softmax(Y_dash)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_dash, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)


train_losses = list()
train_acc = list()
test_losses = list()
test_acc = list()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(1 + epochs):
        batch_X, batch_Y = np.asarray(X_train), np.asarray(Y_train)
        if i%step ==0:
            acc_trn, loss_trn, w, b = sess.run([accuracy, cross_entropy, weights, biases], feed_dict={X: batch_X, Y_: batch_Y})
            acc_tst, loss_tst = sess.run([accuracy, cross_entropy], feed_dict={X: X_test, Y_: Y_test})

            print("#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i,acc_trn,loss_trn,acc_tst,loss_tst))

            train_losses.append(loss_trn)
            train_acc.append(acc_trn)
            test_losses.append(loss_tst)
            test_acc.append(acc_tst)

        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

    save_path = saver.save(sess, "model_mnist.ckpt")

# confusion = confusion_matrix(mnist.test.labels, np.argmax(test_predictions,axis=1))        
title = "Loss Plots"
vis.losses_accuracies_plots(train_losses,train_acc,test_losses, test_acc,title,step	)
