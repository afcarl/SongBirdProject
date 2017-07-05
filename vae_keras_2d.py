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

stack = 5

x_train = x_train.reshape(len(x_train)/stack, stack*x_train.shape[1])
x_test = x_test.reshape(len(x_test)/stack, stack*x_test.shape[1])

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
original_dim = 120*stack
latent_dim = 15
intermediate_dim1 = 90*stack
intermediate_dim2 = 60*stack
intermediate_dim3 = 30*stack
epochs = 30
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h1 = Dense(intermediate_dim1, activation='relu', kernel_initializer=initializers.he_normal())(x)
h2 = Dense(intermediate_dim2, activation='relu', kernel_initializer=initializers.he_normal())(h1)
h3 = Dense(intermediate_dim3, activation='relu', kernel_initializer=initializers.he_normal())(h2)
z_mean = Dense(latent_dim, kernel_initializer=initializers.he_normal())(h3)
z_log_var = Dense(latent_dim, kernel_initializer=initializers.he_normal())(h3)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h3 = Dense(intermediate_dim3, activation='relu', kernel_initializer=initializers.he_normal())
decoder_h2 = Dense(intermediate_dim2, activation='relu', kernel_initializer=initializers.he_normal())
decoder_h1 = Dense(intermediate_dim1, activation='relu', kernel_initializer=initializers.he_normal())
decoder_mean = Dense(original_dim, activation='sigmoid', kernel_initializer=initializers.he_normal())
h_decoded3 = decoder_h3(z)
h_decoded2 = decoder_h2(h_decoded3)
h_decoded1 = decoder_h1(h_decoded2)
x_decoded_mean = decoder_mean(h_decoded1)



# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
    def vae_loss(self, x, x_decoded_mean):
        # xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        mse_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(mse_loss + kl_loss)
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
        epochs=20,
        batch_size=batch_size,
        validation_data=(x_test, x_test))


plt.imshow(x_test[0:100].reshape(500, -1), cmap = 'Greys_r')
plt.show()
res = tmp.predict(x_test[0:100], batch_size=batch_size)
plt.imshow(res.reshape(500, -1), cmap = 'Greys_r')
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
