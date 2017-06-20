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
x_train = dset[0:50000]
x_test = dset[50000:51000]

x_train += 1.0
x_test += 1.0
x_train /= 2.0
x_test /= 2.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras import initializers

batch_size = 10
original_dim = 120
latent_dim = 10
intermediate_dim = 60
epochs = 30
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu', kernel_initializer=initializers.he_normal())(x)
z_mean = Dense(latent_dim, kernel_initializer=initializers.he_normal())(h)
z_log_var = Dense(latent_dim, kernel_initializer=initializers.he_normal())(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu', kernel_initializer=initializers.he_normal())
decoder_mean = Dense(original_dim, activation='sigmoid', kernel_initializer=initializers.he_normal())
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)



# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x_decoded_mean

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
np.random.shuffle(x_train)
np.random.shuffle(x_test)

tmp = Model(x, x_decoded_mean)

# res = tmp.predict(x_test, batch_size=batch_size)
# plt.imshow(res, cmap = 'Greys_r')
# plt.show()
# plt.imshow(x_test, cmap = 'Greys_r')
# plt.show()




vae.fit(x_train,
        shuffle=True,
        epochs=30,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

res = tmp.predict(x_test[0:100], batch_size=batch_size)
plt.imshow(x_test[0:100], cmap = 'Greys_r')
plt.show()
plt.imshow(res, cmap = 'Greys_r')
plt.show()


