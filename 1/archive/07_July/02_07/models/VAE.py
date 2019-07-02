## libs
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

## Keras
from keras.layers import Lambda, Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf

from sklearn.model_selection import train_test_split


class VAE():

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # then z = z_mean + sqrt(var)*eps

    def sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def __init__(self, modelArgs, trainArgs, G, T_array):

        ## MODEL ______________________________________________________________

        ## Multi-layer Perceptron without convolutions__________________________________
        if modelArgs["nn_architecture"] == "mlp":
            ## 1) build encoder model __________________________

            inputs = Input(shape=modelArgs["input_shape"], name='encoder_input')
            x = Dense(128, activation='relu')(inputs)
            x = Dense(64, activation='relu')(x)
            z_mean = Dense(modelArgs["latent_dim"], name='z_mean')(x)
            z_log_var = Dense(modelArgs["latent_dim"], name='z_log_var')(x)

            # use reparameterization trick to push the sampling out as input
            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = Lambda(self.sampling, output_shape=(modelArgs["latent_dim"],), name='z')([z_mean, z_log_var])

            ## 2) build decoder model __________________________

            latent_inputs = Input(shape=(modelArgs["latent_dim"],), name='z_sampling')
            y = Dense(64, activation='relu')(latent_inputs)
            y = Dense(128, activation='relu')(y)
            graph_outputs = Dense(modelArgs["output_shape"], activation='sigmoid')(y)

        ## Convolutional Neural Network_________________________________

        if modelArgs["nn_architecture"] == "2D_conv":

            ## 1) build encoder model____________________________________

            inputs = Input(shape=modelArgs["input_shape"], name='encoder_input')
            x = inputs

            for i in range(2):
                modelArgs['filters'] *= 2
                x = Conv2D(filters=modelArgs['filters'], kernel_size=modelArgs['kernel_size'], activation='relu',
                           strides=2, padding='same')(x)

            # shape info needed to build decoder model
            shape = K.int_shape(x)

            # generate latent vector Q(z|X)
            x = Flatten()(x)
            x = Dense(16, activation='relu')(x)
            z_mean = Dense(modelArgs["latent_dim"], name='z_mean')(x)
            z_log_var = Dense(modelArgs["latent_dim"], name='z_log_var')(x)

            # use reparameterization trick to push the sampling out as input
            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = Lambda(self.sampling, output_shape=(modelArgs["output_shape"],), name='z')([z_mean, z_log_var])

            ## 2) build decoder model____________________________________

            latent_inputs = Input(shape=(modelArgs["latent_dim"],), name='z_sampling')
            x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
            x = Reshape((shape[1], shape[2], shape[3]))(x)

            for i in range(2):
                x = Conv2DTranspose(filters=modelArgs['filters'], kernel_size=modelArgs['kernel_size'],
                                    activation='relu', strides=2, padding='same')(x)
                modelArgs['filters'] //= 2

            graph_outputs = Conv2DTranspose(filters=1, kernel_size=modelArgs['kernel_size'], activation='sigmoid',
                                            padding='same', name='decoder_output')(x)

        ## 2.2) decode growth parameters_________________________________

        # g = Dense(4, activation='relu')(latent_inputs)
        param_outputs = Dense(modelArgs["growth_param"], activation='linear')(latent_inputs)

        ## INSTANTIATE___________________________________

        if modelArgs["param_loss"] == False:
            ## 1) instantiate encoder model
            encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
            encoder.summary()
            # plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

            ## 2) instantiate decoder model
            graph_decoder = Model(latent_inputs, graph_outputs, name='decoder')
            graph_decoder.summary()
            # plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

            ## 3) instantiate VAE model
            outputs = graph_decoder(encoder(inputs)[2])
            vae = Model(inputs, outputs, name='conv_vae')
            # vae.summary()

        if modelArgs["param_loss"]:
            ## 1) instantiate encoder model
            encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
            encoder.summary()
            # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

            ## 2) instantiate decoder model
            graph_decoder = Model(latent_inputs, graph_outputs, name='graph_decoder')
            graph_decoder.summary()
            # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

            ## 2.2) decode growth parameters
            param_decoder = Model(latent_inputs, param_outputs, name='param_decoder')
            param_decoder.summary()

            ## 3) instantiate VAE model
            graph_outputs = graph_decoder(encoder(inputs)[2])
            param_outputs = param_decoder(encoder(inputs)[2])
            outputs = [graph_outputs, param_outputs]

            vae = Model(inputs, outputs, name='vae_graph')
            # vae.summary()

        ## TRAINING ______________________________________

        ## Train and Validation Split _______________________________________________

        x_train, x_test, y_train, y_test = train_test_split(G, T_array, test_size=trainArgs["data_split"],
                                                            random_state=1, shuffle=True)

        def reconstr_loss_func(y_true, y_pred):

            ## RECONSTRUCTION LOSS_______________________

            if trainArgs["loss"] == "mse":

                if modelArgs["nn_architecture"] == "mlp":
                    reconstruction_loss = mse(y_true[0], y_pred[0])
                    reconstruction_loss *= modelArgs["input_shape"]

                if modelArgs["nn_architecture"] == "2D_conv":
                    reconstruction_loss = mse(K.flatten(y_true[0]), K.flatten(y_pred[0]))
                    reconstruction_loss *= modelArgs["input_shape"][0] * modelArgs["input_shape"][1]

            if trainArgs["loss"] == "binary_crossentropy":

                if modelArgs["nn_architecture"] == "mlp":
                    reconstruction_loss = binary_crossentropy(y_true[0], y_pred[0])
                    reconstruction_loss *= modelArgs["input_shape"]

                if modelArgs["nn_architecture"] == "2D_conv":
                    reconstruction_loss = binary_crossentropy(K.flatten(y_true[0]), K.flatten(y_pred[0]))
                    reconstruction_loss *= modelArgs["input_shape"][0] * modelArgs["input_shape"][1]

            ## KL LOSS _____________________________________________

            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5

            ## COMPLETE LOSS __________________________________________________

            reconstr_loss = K.mean(reconstruction_loss + (trainArgs["beta"] * kl_loss))

            return reconstr_loss

        # vae.add_loss(vae_loss)
        # vae_loss = reconstr_loss_func(inputs, outputs[0])

        ## PARAMETER LOSS_______________________

        def param_loss_func(y_true, y_pred):

            param_loss = mse(y_pred[1], y_true[1])

            return param_loss

        if modelArgs["param_loss"] == False:
            vae.compile(optimizer='adam', loss=reconstr_loss_func, metrics=['accuracy'])
            vae.summary()

        if modelArgs["param_loss"]:
            vae.compile(optimizer='adam', loss=[reconstr_loss_func, param_loss_func], metrics=['accuracy'])
            vae.summary()

        ## TRAIN______________________________________________

        # load the autoencoder weights

        if trainArgs["weights"] == "load":

            vae.load_weights("models/weights/vae_mlp_mnist_latent_dim_" + str(modelArgs["latent_dim"]) + ".h5")

        # train the autoencoder

        elif trainArgs["weights"] == "train":

            # Set callback functions to early stop training and save the best model so far
            callbacks = [EarlyStopping(monitor='val_loss', patience=trainArgs["early_stop"]), ModelCheckpoint(
                filepath="models/weights/vae_mlp_mnist_latent_dim_" + str(modelArgs["latent_dim"]) + ".h5",
                save_best_only=True)]

            if modelArgs["param_loss"] == False:
                vae.fit(x_train, [x_train], epochs=trainArgs["epochs"], batch_size=trainArgs["batch_size"],
                        callbacks=callbacks, validation_data=(x_test, [x_test]))
                vae.save_weights("models/weights/vae_mlp_mnist_latent_dim_" + str(modelArgs["latent_dim"]) + ".h5")
                models = (encoder, graph_decoder)

            if modelArgs["param_loss"]:
                vae.fit(x_train, [x_train, y_train], epochs=trainArgs["epochs"], batch_size=trainArgs["batch_size"],
                        callbacks=callbacks, validation_data=(x_test, [x_test, y_test]))
                vae.save_weights("models/weights/vae_mlp_mnist_latent_dim_" + str(modelArgs["latent_dim"]) + ".h5")
                models = (encoder, graph_decoder, param_decoder)

        data = (x_test, y_test)

        self.model = models
        self.data = data