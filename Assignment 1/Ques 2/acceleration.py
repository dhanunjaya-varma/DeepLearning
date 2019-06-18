#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:39:45 2019

@author: dhanunjaya
"""

import tensorflow as tf
import numpy as np

Mass = tf.Variable(np.load('./q2_input/masses.npy'),dtype=tf.float32)
Position = tf.Variable(np.load('./q2_input/positions.npy'),dtype=tf.float32)
Velocity = tf.Variable(np.load('./q2_input/velocities.npy'),dtype=tf.float32)
num_size = 100
switch = tf.Variable(0)
Gravity = tf.Variable(6.67 * (10 ** 5))
threshold= 0.1
delta_time = tf.Variable(10 ** (-4))
graph = tf.get_default_graph()

## Calculate distance (ever particle with every other) tensor(sqrt(square(diff(x)))+ sqrt(square(diff(y)))
## Use Matrix transformations 

r = tf.reduce_sum(Position*Position, 1)
r = tf.reshape(r, [-1, 1])
Distance = r - 2*tf.matmul(Position, tf.transpose(Position)) + tf.transpose(r)
Distance = tf.sqrt(Distance)

## Check if any element in distance matrix is lessthan or equal to given threshold

one_diag=tf.ones(num_size)
#d11=10*d11
zero_diag= tf.zeros([100],tf.float32)
Distance = tf.matrix_set_diag(Distance, one_diag)
switch = tf.cond(tf.size(tf.where(tf.less_equal(Distance, threshold))) > 0, lambda: tf.assign(switch, 1), lambda: tf.assign(switch, 0))
Distance = tf.matrix_set_diag(Distance, zero_diag)

## Calculate 1/cube(r) tensor

Displacement=Distance
Displacement = tf.matrix_set_diag(Displacement, one_diag)
Displacement = tf.reciprocal(Displacement)
Displacement = tf.matrix_set_diag(Displacement, zero_diag)
Displacement=tf.cast(tf.pow(Displacement, 3), dtype = tf.float32)

## Calculate r(x,y) vector, As tensors shapes are different prepare them so that we can apply tensor operations

rw=tf.reshape(Position,[1,-1])
fw=tf.tile(rw,[100,1])
c=tf.tile(Position,[1,100])
g=fw-c

## calculate Mass*1/cube(r)* r vector*Gravity, Diagonal of the resultant matrix will have required acceleation summation

F=tf.tile(Mass,[1,100])
G=tf.transpose(F)
Sigma=G*Displacement  
mr=tf.matmul(Sigma,g)
mrsp=tf.reshape(mr,[100,100,2])
i=tf.transpose(mrsp,perm=[2,1,0])
j=tf.diag_part(i[0,:,:])
k=tf.diag_part(i[1,:,:])
l=tf.transpose(tf.reshape(tf.concat([j,k],0),[2,-1]))
Acc=Gravity*l

## Update Positions and Velocities

Position=tf.assign(Position,Position+(delta_time*Velocity)+((1/2)*(delta_time**2)*Acc))
Velocity=tf.assign(Velocity,Velocity+delta_time*Acc)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

## Variables to Iterate loop
count=0
flag=0

while (flag == 0):
    flag = sess.run(switch)
    print("number of iterations :", count)
    count=count+1
    Updated_Positions =sess.run(Position) 
    Updated_Velocities =sess.run(Velocity)

sess.close()