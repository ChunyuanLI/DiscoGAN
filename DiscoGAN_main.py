from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from data_gmm import GMM_distribution, sample_GMM, plot_GMM
from data_utils import shuffle, iter_data
from tqdm import tqdm

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace


""" parameters """
n_epoch = 1000
batch_size  = 64
dataset_size = 512
input_dim = 2
latent_dim = 2
eps_dim = 2

n_layer_disc = 2
n_hidden_disc = 256
n_layer_gen = 2
n_hidden_gen= 256
n_layer_inf = 2
n_hidden_inf= 256


""" Create directory for results """
result_dir = 'results/DiscoGAN/'
directory = result_dir
if not os.path.exists(directory):
    os.makedirs(directory)


""" Create dataset """

# create X dataset
means = map(lambda x:  np.array(x), [[0, 0],
                                     [2, 2],
                                     [-1, -1],
                                     [1, -1],
                                     [-1, 1]])
means = list(means)
std = 0.1
variances = [np.eye(2) * std for _ in means]

priors = [1.0/len(means) for _ in means]

gaussian_mixture = GMM_distribution(means=means,
                                               variances=variances,
                                               priors=priors)
dataset = sample_GMM(dataset_size, means, variances, priors, sources=('features', ))
save_path = result_dir + 'X_gmm_data.pdf'
plot_GMM(dataset, save_path)

X_np_data= dataset.data['samples']
X_labels = dataset.data['label']

# create Z dataset
means = map(lambda x:  np.array(x), [[-1, -1],[1, 1]])
means = list(means)
std = 0.1
variances = [np.eye(2) * std for _ in means]

priors = [1.0/len(means) for _ in means]

gaussian_mixture = GMM_distribution(means=means,
                                               variances=variances,
                                               priors=priors)
dataset = sample_GMM(dataset_size, means, variances, priors, sources=('features', ))
save_path = result_dir + 'Z_gmm_data.pdf'
plot_GMM(dataset, save_path)

Z_np_data= dataset.data['samples']
Z_labels = dataset.data['label']


# true samples of x and z
X_dataset = X_np_data
Z_dataset = Z_np_data



""" Networks """
def generative_network(z, input_dim, n_layer, n_hidden, eps_dim):
    with tf.variable_scope("generative"):
        h = z
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, input_dim, activation_fn=None, scope="p_x")
    return x


def inference_network(x, latent_dim, n_layer, n_hidden, eps_dim):
    with tf.variable_scope("inference"):
        h = x
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        z = slim.fully_connected(h, latent_dim, activation_fn=None, scope="q_z")
    return z

def data_network_x(x, n_layers=2, n_hidden=256, activation_fn=None):
    """Approximate x log data density."""
    h = tf.concat(x, 1)
    with tf.variable_scope('discriminator_x'):
        h = slim.repeat(h, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d, squeeze_dims=[1])

def data_network_z(z, n_layers=2, n_hidden=256, activation_fn=None):
    """Approximate z log data density."""
    h = tf.concat(z, 1)
    with tf.variable_scope('discriminator_z'):
        h = slim.repeat(h, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d, squeeze_dims=[1])


""" Construct model and training ops """
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
z = tf.placeholder(tf.float32, shape=(batch_size, latent_dim))

# decoder and encoder
p_x = generative_network(z, input_dim , n_layer_gen, n_hidden_gen, eps_dim)
q_z = inference_network(x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)

decoder_logit_x = data_network_x(p_x, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
encoder_logit_x = graph_replace(decoder_logit_x, {p_x: x})
decoder_logit_z = data_network_z(q_z, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
encoder_logit_z = graph_replace(decoder_logit_z, {q_z: z})

encoder_sigmoid_x = tf.nn.softplus(encoder_logit_x)
decoder_sigmoid_x = tf.nn.softplus(decoder_logit_x)
encoder_sigmoid_z = tf.nn.softplus(encoder_logit_z)
decoder_sigmoid_z = tf.nn.softplus(decoder_logit_z)

decoder_loss = decoder_sigmoid_x + decoder_sigmoid_z
encoder_loss = encoder_sigmoid_x + encoder_sigmoid_z

# decoder_loss = decoder_logit_x + decoder_logit_z
# encoder_loss = encoder_logit_x + encoder_logit_z

disc_loss = tf.reduce_mean(  encoder_loss ) - tf.reduce_mean( decoder_loss)

rec_z = inference_network(p_x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim )
cost_z = tf.reduce_mean(tf.pow(rec_z - z, 2))

rec_x = generative_network(q_z, input_dim , n_layer_gen, n_hidden_gen,  eps_dim )
cost_x = tf.reduce_mean(tf.pow(rec_x - x, 2))

adv_loss = tf.reduce_mean(  decoder_loss ) # + tf.reduce_mean( encoder_loss )
gen_loss = 1*adv_loss + 1.*cost_x  + 1.*cost_z

qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference")
pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative")
dvars_x = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_x")
dvars_z = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_z")
opt = tf.train.AdamOptimizer(1e-4, beta1=0.5)
train_gen_op =  opt.minimize(gen_loss, var_list=qvars + pvars)
train_disc_op = opt.minimize(disc_loss, var_list=dvars_x + dvars_z)


""" training """
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


FG = []
FD = []

for epoch in tqdm( range(n_epoch), total=n_epoch):
    X_dataset, Z_dataset= shuffle(X_dataset, Z_dataset)

    for xmb, zmb in iter_data(X_dataset, Z_dataset, size=batch_size):

        for _ in range(1):
            f_d, _ = sess.run([disc_loss, train_disc_op], feed_dict={x: xmb, z:zmb})
        for _ in range(5):
            f_g, _ = sess.run([[adv_loss, cost_x, cost_z], train_gen_op], feed_dict={x: xmb, z:zmb})

        FG.append(f_g)
        FD.append(f_d)


""" plot the results """

n_viz = 1
imz = np.array([]); rmz = np.array([]); imx = np.array([]); rmx = np.array([]);
for _ in range(n_viz):
    for xmb, zmb in iter_data(X_np_data, Z_np_data, size=batch_size):
        temp_imz = sess.run(q_z, feed_dict={x: xmb, z:zmb})
        imz = np.vstack([imz, temp_imz]) if imz.size else temp_imz

        temp_rmz = sess.run(rec_z, feed_dict={x: xmb, z:zmb})
        rmz = np.vstack([rmz, temp_rmz]) if rmz.size else temp_rmz

        temp_imx = sess.run(p_x, feed_dict={x: xmb, z:zmb})
        imx = np.vstack([imx, temp_imx]) if imx.size else temp_imx

        temp_rmx = sess.run(rec_x, feed_dict={x: xmb, z:zmb})
        rmx = np.vstack([rmx, temp_imx]) if rmx.size else temp_rmx

## inferred marginal z
fig_mz, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ll = np.tile(X_labels, (n_viz))
ax.scatter(imz[:, 0], imz[:, 1], c=cm.Set1(ll.astype(float)/input_dim/2.0),
        edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$z_1$'); ax.set_ylabel('$z_2$')
ax.axis('on')
plt.savefig(result_dir + 'inferred_mz.pdf', transparent=True, bbox_inches='tight')

##  reconstruced z
fig_pz, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ll = np.tile(Z_labels, (n_viz))
ax.scatter(rmz[:, 0], rmz[:, 1], c=cm.Set1(ll.astype(float)/input_dim/2.0),
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$z_1$'); ax.set_ylabel('$z_2$')
ax.axis('on')
plt.savefig(result_dir + 'reconstruct_mz.pdf', transparent=True, bbox_inches='tight')

## inferred marginal x
fig_pz, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ll = np.tile(Z_labels, (n_viz))
ax.scatter(imx[:, 0], imx[:, 1], c=cm.Set1(ll.astype(float)/input_dim/2.0),
        edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.axis('on')
plt.savefig(result_dir + 'inferred_mx.pdf', transparent=True, bbox_inches='tight')

##  reconstruced x
fig_mx, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ll = np.tile(X_labels, (n_viz))
ax.scatter(rmx[:, 0], rmx[:, 1], c=cm.Set1(ll.astype(float)/input_dim/2.0),
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.axis('on')
plt.savefig(result_dir + 'reconstruct_mx.pdf', transparent=True, bbox_inches='tight')

## learning curves
fig_curve, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.plot(FD, label="Discriminator")
ax.plot(np.array(FG)[:,0], label="Generator")
ax.plot(np.array(FG)[:,1], label="Reconstruction x")
ax.plot(np.array(FG)[:,2], label="Reconstruction Z")
plt.xlabel('Iteration')
plt.xlabel('Loss')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.axis('on')
plt.savefig(result_dir + 'learning_curves.pdf', bbox_inches='tight')
