import tensorflow as tf

# We want the model to learn: y = 3x
x = tf.constant(2.0)
y_true = tf.constant(6.0)

# Trainable variable (weight)
w = tf.Variable(0.0)

learning_rate = 0.1

print("Starting w:", w.numpy())

for i in range(20):
    with tf.GradientTape() as tape:      # Record all operations
        y_pred = w * x
        loss = (y_true - y_pred) ** 2

    grad = tape.gradient(loss, w) # Compute gradient of loss w.r.t w

    # Update w
    w.assign_sub(learning_rate * grad) 

    print(i, "w =", w.numpy(), "loss =", loss.numpy())

print("Final learned w:", w.numpy())

''' Starting w: 0.0
0 w = 2.4 loss = 36.0
1 w = 2.88 loss = 1.4399996
2 w = 2.976 loss = 0.05759989
3 w = 2.9952 loss = 0.0023039863
4 w = 2.99904 loss = 9.216312e-05
5 w = 2.999808 loss = 3.6872573e-06
6 w = 2.9999616 loss = 1.4734383e-07
7 w = 2.9999924 loss = 5.893753e-09
8 w = 2.9999986 loss = 2.3283064e-10
9 w = 2.9999998 loss = 8.185452e-12
10 w = 3.0 loss = 2.2737368e-13
11 w = 3.0 loss = 0.0
12 w = 3.0 loss = 0.0
13 w = 3.0 loss = 0.0
14 w = 3.0 loss = 0.0
15 w = 3.0 loss = 0.0
16 w = 3.0 loss = 0.0
17 w = 3.0 loss = 0.0
18 w = 3.0 loss = 0.0
19 w = 3.0 loss = 0.0
Final learned w: 3.0
 '''