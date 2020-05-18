import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_random_seed(1)
learning_rate = 0.001  # 0.0001  # GD
num_steps = 5000
display_step = 100

# Process X(t) = b(X(t))dt + sigma(X(t))dW(t) with X0 > 0
x0 = 0  # 1
b = lambda x: - x + x0  # x
sigma = lambda x: tf.pow(tf.abs(x), 1/2)  # x

# P[X>=x] => x =: prob_geq
all_prob_geqs = np.linspace(0, 4, 40)

# Network Parameters
n_hidden_1 = 10  # 1st layer number of neurons
n_hidden_2 = 10  # 2nd layer number of neurons
n_grid_points = 1000


# Create model
def neural_net(x, exit_activation=False):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1_out = tf.nn.tanh(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1_out, weights['h2']), biases['b2'])
    layer_2_out = tf.nn.tanh(layer_2)
    out_layer = tf.matmul(layer_2_out, weights['out'])  # + biases['out']
    if exit_activation:
        return tf.nn.softplus(out_layer)
    else:
        return out_layer


def loss_func(input, NN_func, i_epoch, prob_geq, regularize=True):
    epsilon = lambda x: tf.pow(x, -2)
    gamma = lambda x: tf.add(tf.pow(x, 1.5), 2)  # tf.constant(2.5, dtype=tf.float32)
    dx_phi = tf.gradients(NN_func(input), input)
    loss_unreg = tf.reduce_mean(tf.pow(dx_phi, 2))

    if regularize:
        tf_1 = tf.constant(np.ones((10, 1)), dtype=tf.float32)  # weight vec is of size 10
        tf_0 = tf.constant(np.zeros((10, 1)), dtype=tf.float32)

        reg_1 = gamma(prob_geq) * tf.maximum(prob_geq - tf.reduce_mean(NN_func(tf_1)), 0)
        reg_i = gamma(prob_geq) * (tf.maximum(-epsilon(i_epoch) - tf.reduce_mean(NN_func(tf_0)), 0) +
                                   tf.maximum(tf.reduce_mean(NN_func(tf_0)) - epsilon(i_epoch), 0))
        return loss_unreg + reg_1 + reg_i
    else:
        return loss_unreg


def loss_func_general(b, sigma, x0, input, NN_func, i_epoch, prob_geq, regularize=True):
    # T = 1, X0 = 0 has to be greater 0 !!!
    epsilon = lambda x: tf.pow(x, -2)
    gamma = lambda x: tf.add(tf.pow(x, 1.5), 2)

    dx_phi = tf.gradients(NN_func(input), input)
    loss_unreg = tf.reduce_mean(tf.truediv(tf.pow(dx_phi - b(NN_func(input)), 2), tf.pow(sigma(NN_func(input)), 2)))

    if regularize:
        tf_1 = tf.constant(np.ones((10, 1)), dtype=tf.float32)  # weight vec is of size 10
        tf_0 = tf.constant(np.zeros((10, 1)), dtype=tf.float32)
        tf_x0 = tf.constant(np.ones((10, 1)) * x0, dtype=tf.float32)

        reg_1 = gamma(prob_geq) * tf.maximum(prob_geq - tf.reduce_mean(NN_func(tf_1)), 0)
        reg_i = gamma(prob_geq) * (tf.maximum(-(epsilon(i_epoch) + tf_x0) - tf.reduce_mean(NN_func(tf_0)), 0) +
                                   tf.maximum(tf.reduce_mean(NN_func(tf_0)) - (epsilon(i_epoch) + tf_x0), 0))

        return loss_unreg + reg_1 + reg_i
    else:
        return loss_unreg


class opt:
    def __init__(self):
        self.lr = 0.0001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.f = np.sqrt(1 - self.beta2) / (1 - self.beta1)
        self.eps = 1e-5
        self.m = {key: tf.zeros_like(val) for key, val in param_dict.items()}
        self.v = {key: tf.zeros_like(val) for key, val in param_dict.items()}

    def adam_tf(self, C, dL, dC):
        dTheta = {key: tf.zeros_like(val) for key, val in param_dict.items()}
        for key, param in param_dict.items():
            m_new = self.beta1 * tf.reshape(self.m[key], (-1, 1)) + (1-self.beta2) * tf.reshape(dL[key], (-1, 1))
            v_new = self.beta2 * tf.reshape(self.v[key], (-1, 1)) + (1-self.beta2) * tf.pow(tf.reshape(dL[key],
                                                                                                       (-1, 1)), 2)

            A11 = self.lr*self.f*tf.diag(tf.squeeze(tf.sqrt(v_new) + self.eps))
            A21 = tf.reshape(dC[key], (-1, A11.shape[1]))
            A12 = tf.transpose(A21)
            A22 = tf.zeros((A21.shape[0], A12.shape[1]))

            rhs = tf.concat([-m_new, -C], 0)
            matrix = tf.concat([tf.concat([A11, A12], 1), tf.concat([A21, A22], 1)], 0)
            print(matrix)
            out = tf.linalg.solve(matrix, rhs, adjoint=False, name=None)

            dTheta[key] = tf.reshape(out[:-C.shape[0].value], param.shape)
            self.m[key] = tf.reshape(m_new, param.shape)
            self.v[key] = tf.reshape(v_new, param.shape)

        return dTheta


final_losses = np.array([])
for single_prob_geq in all_prob_geqs:
    prob_geq = tf.constant(single_prob_geq, dtype=tf.float32)
    X = tf.placeholder("float", [None, 1])
    i_epoch = tf.placeholder(tf.float32)

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([1, n_hidden_1]), name='h1'),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2'),
        'out': tf.Variable(tf.random_normal([n_hidden_2, 1]), name='hout')
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
        # 'out': tf.Variable(tf.random_normal([1]), name='bout')
    }

    param_dict = {**weights, **biases}

    # loss_op = loss_func(X, neural_net, i_epoch, prob_geq)
    # unreg_loss_op = loss_func(X, neural_net, i_epoch, prob_geq, regularize=False)

    neural_net_positive = lambda x: neural_net(x, exit_activation=True)
    # loss_op = loss_func_general(b, sigma, x0, X, neural_net_positive, i_epoch, prob_geq)
    # unreg_loss_op = loss_func_general(b, sigma, x0, X, neural_net_positive, i_epoch, prob_geq, regularize=False)
    loss = loss_func(X, neural_net_positive, None, prob_geq, regularize=False)
    constraints = tf.concat([neural_net_positive(tf.constant([[0.]])),
                             neural_net_positive(tf.constant([[1.]])) - prob_geq], 0)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # gradient_step = optimizer.compute_gradients(loss, tf.trainable_variables())
    gradient_step, variables = zip(*optimizer.compute_gradients(loss))
    gradient_step, _ = tf.clip_by_global_norm(gradient_step, 5.0)

    # gradient_step_constraints = optimizer.compute_gradients(constraints, tf.trainable_variables())
    C_gradient_step0, _ = zip(*optimizer.compute_gradients(constraints[0, 0]))
    # C_gradient_step0, _ = tf.clip_by_global_norm(C_gradient_step0, 0.5)
    C_gradient_step1, _ = zip(*optimizer.compute_gradients(constraints[1, 0]))
    # C_gradient_step1, _ = tf.clip_by_global_norm(C_gradient_step1, 0.5)
    C_gradient_step = zip(C_gradient_step0, C_gradient_step1)

    gradient_dict = {key: param for key, param in zip(param_dict.keys(), gradient_step)}
    C_gradient_dict = {}  # {key: tf.concat([param0, param1], 1) for key, (param0, param1) in zip(param_dict.keys(), list(C_gradient_step))}
    for key, (param0, param1) in zip(param_dict.keys(), C_gradient_step):
        if len(param0.shape) == 1:
            C_gradient_dict[key] = tf.concat([tf.expand_dims(param0, 1), tf.expand_dims(param1, 1)], 1)
        else:
            C_gradient_dict[key] = tf.concat([param0, param1], 1)

    update_gradients = opt().adam_tf(constraints, gradient_dict, C_gradient_dict)

    optimize = optimizer.apply_gradients(zip(update_gradients.values(), variables))

    # train_op = optimizer.minimize(loss_op)

    # l1_reg = sum([0.001*tf.reduce_sum(tf.abs(weights[w])) for w in weights.keys()])
    # train_op = optimizer.minimize(loss_op + l1_reg)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        for step in range(1, num_steps + 1):
            x_grid = np.reshape(np.linspace(-0.0, 1.0, n_grid_points), (n_grid_points, 1))  # slightly bigger interval?
            # x_unif = np.random.rand(n_grid_points, 1)
            # x_unif = np.random.uniform(-0.25, 1.25, (n_grid_points, 1))

            # Run optimization op (backprop)
            # sess.run(train_op, feed_dict={X: x_grid, i_epoch: step})
            # tf_loss = sess.run(loss, feed_dict={X: x_grid})

            # gradients = sess.run(gradient_step, feed_dict={X: x_grid})
            # C_gradients = sess.run(C_gradient_step, feed_dict={X: x_grid})
            sess.run(update_gradients, feed_dict={X: x_grid})
            # sess.run(optimize, feed_dict={X: x_grid})

            print('stop')


        # final_losses = np.append(final_losses, sess.run(unreg_loss_op, feed_dict={X: x_grid}))
        final_losses = np.append(final_losses, sess.run(loss, feed_dict={X: x_grid}))
    print('x = %f Optimized!' % single_prob_geq)

print("Optimization Finished!")

plt.plot(all_prob_geqs, final_losses, '*', label='neural_net_approx')
# plt.plot(all_prob_geqs, np.power(all_prob_geqs, 2), '-', label=r'$x^2$')
plt.legend()
plt.title('Hidden: {}/{}, MSE: {}'.format(n_hidden_1, n_hidden_2,
                                          np.mean(np.power(final_losses - np.power(all_prob_geqs, 2), 2))))
plt.show()
