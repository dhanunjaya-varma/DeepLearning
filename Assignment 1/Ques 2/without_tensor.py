#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:18:33 2019

"""

import tensorflow as tf
import numpy as np


def acceleration(masses, pos, threshold):
    G = -6.67 * (10 ** 5)
    flag = 0
    iterator = np.size(mas)
    
    pos_c1 = pos[:, 0] 
    pos_c2 = pos[:, 1] 
    diff_pos_c1 = np.subtract.outer(pos_c1, pos_c1)
    diff_pos_c2 = np.subtract.outer(pos_c2, pos_c2)
    
    square_outer_c1 = np.square(diff_pos_c1)
    square_outer_c2 = np.square(diff_pos_c2)
    dist = np.sqrt(square_outer_c1 + square_outer_c2) 
    np.fill_diagonal(dist, 1000)
    for i in range(iterator):
        for j in range(iterator):
            if dist[i][j] <= threshold:
                flag = 1
                break
    dist = np.reciprocal(dist)
    dist = G * np.power(dist, 3)
    np.fill_diagonal(dist, 0)
    dist_c1 = np.multiply(dist, diff_pos_c1)
    dist_c2 = np.multiply(dist, diff_pos_c2)
    acc_c1 = np.matmul(dist_c1, np.reshape(masses, (np.size(masses), 1)))
    acc_c2 = np.matmul(dist_c2, np.reshape(masses, (np.size(masses), 1)))
    acc_c1 = np.reshape(acc_c1, (np.size(acc_c1), 1))
    acc_c2 = np.reshape(acc_c2, (np.size(acc_c2), 1))
    acc = np.concatenate((acc_c1, acc_c2), 1)
    return acc, flag


def update_system(mas, pos, vec):
    time_step = 10 ** (-4)
    threshold = 0.1
    p=pos
    v=vec
    updated_pos = []
    updated_vec= []
    updated_pos.append(p)
    updated_vec.append(v)
    flag = 0
    counter = 0
    while flag == 0:
        acc, flag = acceleration(mas, p, threshold)
        p = p + time_step * v + (0.5) * (time_step ** 2) * acc
        v = v + time_step * acc
        updated_pos.append(p)
        updated_vec.append(v)
        counter += 1
        print("number of iterations :", counter)
    return updated_pos, updated_vec

mas = np.load('./q2_input/masses.npy')
pos = np.load('./q2_input/positions.npy')
vec = np.load('./q2_input/velocities.npy')
updated_pos, updated_vec = update_system(mas, pos, vec) 
