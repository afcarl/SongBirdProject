from __future__ import print_function
import argparse
import pdb
import h5py
import numpy as np
import matplotlib.pyplot as plt
# from scipy.misc import imsave
path2dat = '/media/songbird/Data/deep_learn_data/'
filnam = path2dat+'song_data.hdf5'

f = h5py.File(filnam,'r')

dset = f[u'songdata']
x_train = dset[0:100000]
x_test = dset[100000:110000]

x_train += 1.0
x_test += 1.0
x_train /= 2.0
x_test /= 2.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

stack = 8

x_train = x_train.reshape(len(x_train)/stack, stack*x_train.shape[1])
x_test = x_test.reshape(len(x_test)/stack, stack*x_test.shape[1])

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras import initializers
import keras as keras

batch_size = 10
original_dim = 120*stack
latent_dim = 15*stack
intermediate_dim1 = 90*stack
intermediate_dim2 = 60*stack
intermediate_dim3 = 30*stack
epochs = 30
epsilon_std = 1.0



latent_dim = 6
x = Input(batch_shape=(batch_size, original_dim))

fe1 = Dense(intermediate_dim1, activation='tanh', kernel_initializer=initializers.he_normal())(x)
fe1b = keras.layers.normalization.BatchNormalization()(fe1)
fe2 = Dense(intermediate_dim2, activation='tanh', kernel_initializer=initializers.he_normal())(fe1b)
fe3 = Dense(intermediate_dim3, activation='tanh', kernel_initializer=initializers.he_normal())(fe2)
fe3b = keras.layers.normalization.BatchNormalization()(fe3)

z_mean = Dense(latent_dim, kernel_initializer=initializers.he_normal())(fe3b)
z_log_var = Dense(latent_dim, kernel_initializer=initializers.he_normal())(fe3b)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
fd1 = Dense(intermediate_dim3, activation='tanh', kernel_initializer=initializers.he_normal())(z)
fd1b = keras.layers.normalization.BatchNormalization()(fd1)
fd2 = Dense(intermediate_dim2, activation='tanh', kernel_initializer=initializers.he_normal())(fd1b)
fd3 = Dense(intermediate_dim1, activation='tanh', kernel_initializer=initializers.he_normal())(fd2)
fd3b = keras.layers.normalization.BatchNormalization()(fd3)

fd4_mu = Dense(original_dim, activation='tanh', kernel_initializer=initializers.he_normal())(fd3b)
fd4_sigma = Dense(1, kernel_initializer=initializers.he_normal())(fd3b)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
    def vae_loss(self, x, x_decoded_mean, x_decoded_var):
        # xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        mse_loss = (-(0.4 + 0.5 * x_decoded_var) -
                      0.5 * (K.mean((x - x_decoded_mean)**2, axis = -1) / K.exp(x_decoded_var)))
        # invsig = - 0.5 / K.exp(x_decoded_var)
        # mse_loss = metrics.mean_squared_error(x, x_decoded_mean) * invsig - 0.5 * x_decoded_var - 0.4
        kl_loss = + 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(-1*mse_loss - kl_loss)
    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        x_decoded_var = inputs[2]
        loss = self.vae_loss(x, x_decoded_mean, x_decoded_var)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x_decoded_mean

y = CustomVariationalLayer()([x, fd4_mu, fd4_sigma])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
np.random.shuffle(x_train)
np.random.shuffle(x_test)

tmp = Model(x, fd4_mu)
tmp2 = Model(x, fd4_sigma)

# res = tmp.predict(x_test, batch_size=batch_size)
# plt.imshow(res, cmap = 'Greys_r')
# plt.show()
# plt.imshow(x_test, cmap = 'Greys_r')
# plt.show()




hist = vae.fit(x_train,
        shuffle=True,
        epochs=10,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

plt.plot(hist.history['loss'][1:])
plt.show()

# plt.imshow(x_test[0:100].reshape(100*stack, -1), cmap = 'Greys_r')
# plt.show()
samples = 10
mean = tmp.predict(x_test[0:samples], batch_size=batch_size)
var = tmp2.predict(x_test[0:samples], batch_size=batch_size)
res = mean + np.exp(var/2)*np.random.normal(size=(samples, original_dim), loc=0.,scale=epsilon_std)
plt.imshow(mean.reshape(samples*stack, -1), cmap = 'Greys_r')
plt.show()




decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)



samples = np.random.normal(0,1,(100,latent_dim))
outs = []

for i in samples:
    x_decoded = generator.predict(i.reshape(1,latent_dim))
    outs.append(x_decoded)

out = np.asarray(outs)
out = out.reshape((len(samples), original_dim))
plt.imshow(out, cmap = 'Greys_r')
plt.show()
