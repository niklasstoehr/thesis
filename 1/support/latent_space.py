from matplotlib import pylab as plt
from sklearn.decomposition import PCA
import os
import numpy as np
import scipy
import seaborn as sns



def vis2D(analyzeArgs, modelArgs, models, data, batch_size=128, model_name="vae_graph"):
    """Plots labels and data as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    if modelArgs["param_loss"]:
        encoder, graph_decoder, param_decoder = models  # trained models
    else:
        encoder, graph_decoder = models  # trained models

    if modelArgs["latent_dim"] > 1:

        x_test, y_test = data

        ## ENCODER - 2D Digit Classes _________________________________________

        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)

        ## DIMENSIONALITY REDUCTION _______________________

        pca = PCA(n_components=2)
        projected_z = pca.fit_transform(z_mean)

        ## toDO: add t-SNE

        plt.figure(figsize=(12, 10))
        # plt.scatter(projected_z[:, 0], projected_z[:, 1], c=y_test[:0], edgecolor='none', alpha=0.6)
        plt.scatter(projected_z[:, 0], projected_z[:, 1], edgecolor='none', alpha=0.6)
        plt.xlabel('projected  z_0')
        plt.ylabel('projected  z_1')
        # plt.colorbar()

        if analyzeArgs["save_plots"] == True:
            os.makedirs(model_name, exist_ok=True)
            filename = os.path.join(model_name, "vae_mean.png")
            plt.savefig(filename)

    else:
        print("latent_dim needs to be larger than 1")






def visDistr(modelArgs, analyzeArgs, models, data, batch_size):

    if modelArgs["param_loss"]:
        encoder, graph_decoder, param_decoder = models  # trained models
    else:
        encoder, graph_decoder = models  # trained models

    x_test, y_test = data

    # display a 2D plot of the digit classes in the latent space
    z_mean, z_log_var, z = encoder.predict(x_test, batch_size)

    ## Plot Difference Plot _______________________________

    normal = np.random.gumbel(0.0, 1.0, 100000)
    kde_normal = scipy.stats.gaussian_kde(normal)

    col_titles = ['z_{}'.format(col) for col in range(z_mean.shape[1])]

    for i in range(1, modelArgs["latent_dim"] + 1):
        fig = plt.subplot(1, modelArgs["latent_dim"] + 1, i)

        plt.xlabel('x')
        plt.ylabel('y')
        grid = np.linspace(-4, 4, 20)
        kde_z = scipy.stats.gaussian_kde(z_mean[:, i - 1])

        plt.plot(grid, kde_normal(grid), label="Gaussian prior", color='purple', linestyle='dashed',
                 markerfacecolor='blue', linewidth=6)
        plt.plot(grid, kde_z(grid), label="z", color='midnightblue', markerfacecolor='blue', linewidth=6)
        plt.plot(grid, kde_normal(grid) - kde_z(grid), label="difference", color='steelblue', linewidth=6)

    ## Plot Joint Distribution Plot _______________________________

    ## outline how much learned variables deviates from normal distribution

    if modelArgs["latent_dim"] > 1:
        g = sns.jointplot(z_mean[:, analyzeArgs["z"][0]], z_mean[:, analyzeArgs["z"][1]], kind="kde", space=0)
        plt.show()