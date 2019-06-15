from preprocessing import reconstruct_adjacency, unpad_matrix

import networkx as nx
import numpy as np

from matplotlib import pylab as plt
import os



## apply decoder and generate data

def generate_single(analyzeArgs, modelArgs, dataArgs, models, color_map):

    encoder, decoder = models  # trained models
    print("latent dimensions:", modelArgs["latent_dim"])

    z_sample = np.zeros(modelArgs["latent_dim"])
    z_sample = np.reshape(z_sample, (1, modelArgs["latent_dim"]))

    for i, dim in enumerate(analyzeArgs["z"]):
        z_sample[0][dim] = analyzeArgs["activations"][i]

    x_decoded = decoder.predict(z_sample)

    ## reconstruct upper triangular adjacency matrix
    reconstructed_a = reconstruct_adjacency(x_decoded, dataArgs["clip"], dataArgs["diag_offset"])
    reconstructed_a = unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"])

    ## reconstruct graph
    g = nx.from_numpy_matrix(reconstructed_a)
    # reconstructed_a = nx.adjacency_matrix(g).todense()

    nx.draw(g, node_color=color_map)








## DECODER - Latent Space Interpolation____________________________

def generate_manifold(analyzeArgs, modelArgs, dataArgs, models, data, color_map, batch_size=128):
    print("latent dimensions:", modelArgs["latent_dim"])

    encoder, decoder = models  # trained models
    x_test, y_test = data

    # display a 2D plot of the digit classes in the latent space
    z_mean, z_log_var, z = encoder.predict(x_test, batch_size)

    ## Latent Space Dimension is 1 ______________________

    if modelArgs["latent_dim"] == 1:

        ## 1) create adjacency plots__________________________________________

        # display a 30x30 2D manifold of digits
        n = dataArgs["n_max"]  # number of nodes
        figure = np.zeros((1 * n, analyzeArgs["size_of_manifold"] * n))

        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        if analyzeArgs["sample"] == "z":
            grid_x = np.sort(np.random.normal(np.mean(z_mean[:, analyzeArgs["z"][0]]),
                                              np.mean(np.exp(z_log_var[:, analyzeArgs["z"][0]])),
                                              analyzeArgs["size_of_manifold"]))
        elif analyzeArgs["sample"] == "range":
            grid_x = np.linspace(analyzeArgs["act_range"][0], analyzeArgs["act_range"][1],
                                 analyzeArgs["size_of_manifold"])
        elif analyzeArgs["sample"] == "normal":
            grid_x = np.sort(
                np.random.normal(np.mean(z_mean[:, analyzeArgs["z"][0]]), 1, analyzeArgs["size_of_manifold"]))

        z_sample = np.zeros(modelArgs["latent_dim"])
        z_sample = np.reshape(z_sample, (1, modelArgs["latent_dim"]))

        ## 2) create graph plots_______________________________________________

        fig, axs = plt.subplots(1, analyzeArgs["size_of_manifold"], figsize=(10, 10))
        # fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()

        for j, xi in enumerate(grid_x):

            z_sample[0][0] = xi ** analyzeArgs["act_scale"]
            x_decoded = decoder.predict(z_sample)

            ## reconstruct upper triangular adjacency matrix
            reconstructed_a = reconstruct_adjacency(x_decoded, dataArgs["clip"], dataArgs["diag_offset"])

            ## 1) create adjacency plot_____________________________

            figure[0:n, j * n: (j + 1) * n] = reconstructed_a

            ## 2) create graph plot_____________________________

            # reconstruct graph
            reconstructed_a = unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"])
            g = nx.from_numpy_matrix(reconstructed_a)

            if len(g) > 0:
                reconstructed_a = nx.adjacency_matrix(g).todense()

            # compute index for the subplot, and set this subplot as current
            jx = np.unravel_index(j, axs.shape)
            plt.sca(axs[jx])

            nx.draw(g, node_size=10, node_color=color_map)
            axs[jx].set_axis_off()
            axs[jx].set(ylabel='z_0')

        start_range = n // 2
        end_range = (analyzeArgs["size_of_manifold"] - 1) * n + start_range + 1
        pixel_range = np.arange(start_range, end_range, n)
        sample_range_x = np.round(grid_x, 1)

        # Plot_____________________________

        plt.figure(figsize=(15, 300))
        plt.xticks(pixel_range, sample_range_x)
        plt.xlabel("z_0", fontweight='bold')
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

        if analyzeArgs["save_plots"] == True:
            filename = os.path.join(model_name, "digits_over_latent.png")
            plt.savefig(filename)

            ## Latent Space Dimension is 2 ______________________

    if modelArgs["latent_dim"] == 2:

        ## 1) create adjacency plots_______________________________________________

        # display a 30x30 2D manifold of digits
        n = dataArgs["n_max"]  # number of nodes
        figure = np.zeros((analyzeArgs["size_of_manifold"] * n, analyzeArgs["size_of_manifold"] * n))

        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        if analyzeArgs["sample"] == "z":
            grid_x = np.sort(np.random.normal(np.mean(z_mean[:, analyzeArgs["z"][0]]),
                                              np.mean(np.exp(z_log_var[:, analyzeArgs["z"][0]])),
                                              analyzeArgs["size_of_manifold"]))
            grid_y = np.sort(np.random.normal(np.mean(z_mean[:, analyzeArgs["z"][1]]),
                                              np.mean(np.exp(z_log_var[:, analyzeArgs["z"][1]])),
                                              analyzeArgs["size_of_manifold"]))
        elif analyzeArgs["sample"] == "range":
            grid_x = np.linspace(analyzeArgs["act_range"][0], analyzeArgs["act_range"][1],
                                 analyzeArgs["size_of_manifold"])
            grid_y = np.linspace(analyzeArgs["act_range"][0], analyzeArgs["act_range"][1],
                                 analyzeArgs["size_of_manifold"])[::-1]  ## revert
        elif analyzeArgs["sample"] == "normal":
            grid_x = np.sort(
                np.random.normal(np.mean(z_mean[:, analyzeArgs["z"][0]]), 1, analyzeArgs["size_of_manifold"]))
            grid_y = np.sort(
                np.random.normal(np.mean(z_mean[:, analyzeArgs["z"][1]]), 1, analyzeArgs["size_of_manifold"]))

        ## 2) create graph plots_______________________________________________

        fig, axs = plt.subplots(analyzeArgs["size_of_manifold"], analyzeArgs["size_of_manifold"], figsize=(8, 8))
        # fig.subplots_adjust(hspace = .5, wspace=.001)

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):

                xi_value = xi ** analyzeArgs["act_scale"]
                yi_value = yi ** analyzeArgs["act_scale"]

                z_sample = np.array([[xi_value, yi_value]])
                x_decoded = decoder.predict(z_sample)

                ## reconstruct upper triangular adjacency matrix
                reconstructed_a = reconstruct_adjacency(x_decoded, dataArgs["clip"], dataArgs["diag_offset"])

                ## 1) create adjacency plots_____________________________________

                figure[i * n: (i + 1) * n,
                j * n: (j + 1) * n] = reconstructed_a

                ## 2) create graph plot_____________________________

                ## reconstruct graph
                reconstructed_a = unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"])
                g = nx.from_numpy_matrix(reconstructed_a)

                if len(g) > 0:
                    reconstructed_a = nx.adjacency_matrix(g).todense()

                # compute index for the subplot, and set this subplot as current
                plt.sca(axs[i, j])
                nx.draw(g, node_size=10, node_color=color_map)
                axs[i, j].set_axis_off()

        start_range = n // 2
        end_range = (analyzeArgs["size_of_manifold"] - 1) * n + start_range + 1
        pixel_range = np.arange(start_range, end_range, n)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)

        # Plot_____________________________

        plt.figure(figsize=(10, 10))
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z_0", fontweight='bold')
        plt.ylabel("z_1", fontweight='bold')
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

        if analyzeArgs["save_plots"] == True:
            filename = os.path.join(model_name, "digits_over_latent.png")
            plt.savefig(filename)

            ## Latent Space Dimension is larger than 2 ______________________

    if modelArgs["latent_dim"] > 2:

        z_sample = np.zeros(modelArgs["latent_dim"])
        z_sample = np.reshape(z_sample, (1, modelArgs["latent_dim"]))

        ## 1) create adjacency plots_______________________________________________

        # display a 30x30 2D manifold of digits
        n = dataArgs["n_max"]  # number of nodes
        figure = np.zeros((analyzeArgs["size_of_manifold"] * n, analyzeArgs["size_of_manifold"] * n))

        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        if analyzeArgs["sample"] == "z":
            grid_x = np.sort(np.random.normal(np.mean(z_mean[:, analyzeArgs["z"][0]]),
                                              np.mean(np.square(np.exp(z_log_var[:, analyzeArgs["z"][0]]))),
                                              analyzeArgs["size_of_manifold"]))
            grid_y = np.sort(np.random.normal(np.mean(z_mean[:, analyzeArgs["z"][1]]),
                                              np.mean(np.square(np.exp(z_log_var[:, analyzeArgs["z"][1]]))),
                                              analyzeArgs["size_of_manifold"]))
        elif analyzeArgs["sample"] == "range":
            grid_x = np.linspace(analyzeArgs["act_range"][0], analyzeArgs["act_range"][1],
                                 analyzeArgs["size_of_manifold"])
            grid_y = np.linspace(analyzeArgs["act_range"][0], analyzeArgs["act_range"][1],
                                 analyzeArgs["size_of_manifold"])[::-1]  ## revert
        elif analyzeArgs["sample"] == "normal":
            grid_x = np.sort(
                np.random.normal(np.mean(z_mean[:, analyzeArgs["z"][0]]), 1, analyzeArgs["size_of_manifold"]))
            grid_y = np.sort(
                np.random.normal(np.mean(z_mean[:, analyzeArgs["z"][1]]), 1, analyzeArgs["size_of_manifold"]))

        ## 2) create graph plots_______________________________________________

        fig, axs = plt.subplots(analyzeArgs["size_of_manifold"], analyzeArgs["size_of_manifold"], figsize=(10, 10))
        # fig.subplots_adjust(hspace = .5, wspace=.001)

        ## fill unobserved dimensions with mean of latent variable dimension
        for dim in range(0, len(z_sample[0])):
            z_sample[0][dim] = np.mean(z_mean[:, dim])

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):

                z_sample[0][analyzeArgs["z"][0]] = xi ** analyzeArgs["act_scale"]
                z_sample[0][analyzeArgs["z"][1]] = xi ** analyzeArgs["act_scale"]
                x_decoded = decoder.predict(z_sample)

                ## reconstruct upper triangular adjacency matrix
                reconstructed_a = reconstruct_adjacency(x_decoded, dataArgs["clip"], dataArgs["diag_offset"])

                ## 1) create adjacency plot_____________________________

                figure[i * n: (i + 1) * n,
                j * n: (j + 1) * n] = reconstructed_a

                ## 2) create graph plot_____________________________

                ## reconstruct graph
                reconstructed_a = unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"])
                g = nx.from_numpy_matrix(reconstructed_a)
                if len(g) > 0:
                    reconstructed_a = nx.adjacency_matrix(g).todense()

                # compute index for the subplot, and set this subplot as current
                plt.sca(axs[i, j])
                nx.draw(g, node_size=10, node_color=color_map)
                axs[i, j].set_axis_off()

        start_range = n // 2
        end_range = (analyzeArgs["size_of_manifold"] - 1) * n + start_range + 1
        pixel_range = np.arange(start_range, end_range, n)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)

        # Plot_____________________________

        plt.figure(figsize=(10, 10))
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z_0", fontweight='bold')
        plt.ylabel("z_1", fontweight='bold')
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

        if analyzeArgs["save_plots"] == True:
            filename = os.path.join(model_name, "digits_over_latent.png")
            plt.savefig(filename)

        ## Generate Data _______________________________________

        # range, normal, z
