# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize((loss))

init = tf.global_variables_initializer()

# 创建session, 进行参数初始化
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(weights), sess.run(biases))


matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

state = tf.Variable(0, name="counter")
print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

"""TF placeholder
placeholder 是 Tensorflow 中的占位符，暂时储存变量.
Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).
"""

input1 = tf.placeholder(dtype=tf.float32)
input2 = tf.placeholder(dtype=tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [3.], input2: [5.]}))
