import tensorflow as tf
import visual as vis
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

weights = list()
biases  = list()

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
    


epochs = 5000
step = 100
batch = 100

mnist = read_data_sets("MNIST", one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10]) # correct labels

XX = tf.reshape(X, [-1, 784])

Y = FC(XX, 200, 'sigmoid', True, False)
Y = FC(Y, 100, 'sigmoid', True, False)
Y = FC(Y, 60, 'sigmoid', True, False)
Y = FC(Y, 30, 'sigmoid', True, False)
Y_dash = FC(Y, 10, 'sigmoid', True, True)
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
        batch_X, batch_Y = mnist.train.next_batch(batch)
        # print (batch_Y[0])
        if i%step ==0:
            acc_trn, loss_trn, w, b = sess.run([accuracy, cross_entropy, weights, biases], feed_dict={X: batch_X, Y_: batch_Y})
            acc_tst, loss_tst = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})

            # print (sess.run(correct_prediction, feed_dict={X: batch_X, Y_: batch_Y}))

            print("#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i,acc_trn,loss_trn,acc_tst,loss_tst))

            train_losses.append(loss_trn)
            train_acc.append(acc_trn)
            test_losses.append(loss_tst)
            test_acc.append(acc_tst)

        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

    save_path = saver.save(sess, "model_mnist.ckpt")
  
# confusion = confusion_matrix(mnist.test.labels, np.argmax(test_predictions,axis=1))        
title = "Loss Plots1"
vis.losses_accuracies_plots(train_losses,train_acc,test_losses, test_acc,title,step	)
