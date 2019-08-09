from support.preprocessing import reconstruct_adjacency, unpad_matrix, sort_adjacency, pad_matrix

import sys
import networkx as nx
from networkx.generators import random_graphs
from networkx.generators import classic
import numpy as np

from matplotlib import pylab as plt
import os


def decode_param(analyzeArgs, dataArgs, scaler, x_decoded):
    ## generate graph from generative parameters
    x_decoded = scaler.inverse_transform(x_decoded)
    x_decoded = np.squeeze(x_decoded)

    ## ensure data matches range
    if analyzeArgs["graph_type"] == "Complete":
        n_gen = np.clip(int(x_decoded), 1, dataArgs["n_max"] - 1)
        g = classic.complete_graph(n_gen)

        params = ("n")
        y_pos = np.arange(len(params))
        param_values = [n_gen / dataArgs["n_max"]]

        return g, y_pos, params, param_values

    ## ensure data matches range
    if analyzeArgs["graph_type"] == "Tree":
        b_gen = np.clip(int(x_decoded[0]), 1, dataArgs["n_max"] - 1)
        h_gen = np.clip(int(x_decoded[1]), 1, dataArgs["n_max"] - 1)
        g = classic.balanced_tree(b, h)

        params = ("b", "h")
        y_pos = np.arange(len(params))
        param_values = [b_gen, h_gen]

        return g, y_pos, params, param_values

    ## ensure data matches range
    if analyzeArgs["graph_type"] == "ER":
        n_gen = np.clip(int(x_decoded[0]), 1, dataArgs["n_max"] - 1)
        p_gen = np.clip(x_decoded[1], 0, 1)
        g = random_graphs.erdos_renyi_graph(n_gen, p_gen, seed=None, directed=False)

        params = ("n", "p")
        y_pos = np.arange(len(params))
        param_values = [n_gen / dataArgs["n_max"], p_gen]

        return g, y_pos, params, param_values

    ## ensure data matches range
    if analyzeArgs["graph_type"] == "PA":
        n_gen = np.clip(int(x_decoded[0]), 2, dataArgs["n_max"] - 1)
        e_gen = np.clip(int(x_decoded[1]), 1, n_gen - 1)
        g = random_graphs.barabasi_albert_graph(n_gen, e_gen, seed=None)

        params = ("n", "e")
        y_pos = np.arange(len(params))
        param_values = [n_gen / dataArgs["n_max"], e_gen / n_gen]

        return g, y_pos, params, param_values

    ## ensure data matches range
    if analyzeArgs["graph_type"] == "HK":
        n_gen = np.clip(int(x_decoded[0]), 1, dataArgs["n_max"] - 1)
        e_gen = np.clip(int(x_decoded[1]), 1, n_gen)
        p_gen = np.clip(x_decoded[2], 0, 1)
        g = random_graphs.powerlaw_cluster_graph(n_gen, e_gen, p_gen, seed=None)

        params = ("n", "e", "p")
        y_pos = np.arange(len(params))
        param_values = [n_gen / dataArgs["n_max"], e_gen / n_gen, p_gen]

        return g, y_pos, params, param_values

    ## ensure data matches range
    if analyzeArgs["graph_type"] == "SW":
        n_gen = np.clip(int(x_decoded[0]), 1, dataArgs["n_max"] - 1)
        k_gen = np.clip(int(x_decoded[1]), 0, n_gen - 1)
        p_gen = np.clip(x_decoded[2], 0, 1)
        g = random_graphs.newman_watts_strogatz_graph(n_gen, k_gen, p_gen, seed=None)  # no edges are removed

        params = ("n", "k", "p")
        y_pos = np.arange(len(params))
        param_values = [n_gen / dataArgs["n_max"], k_gen / n_gen, p_gen]

        return g, y_pos, params, param_values



def generate_param_graph_manifold(analyzeArgs, modelArgs, dataArgs, models, data, color_map, batch_size, scaler):
    # DECODER - Latent Space Interpolation____________________________

    print("latent dimensions:", modelArgs["latent_dim"])

    if modelArgs["param_loss"] == False:
        sys.exit("modelArgs[param_loss] should be True")
    else:
        encoder, graph_decoder, param_decoder = models  # trained models

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
            x_decoded = param_decoder.predict(z_sample)

            g, y_pos, params, param_values = decode_param(analyzeArgs, dataArgs, scaler, x_decoded)

            ## convert graph to adjacency
            g, reconstructed_a = sort_adjacency(g)
            reconstructed_a = pad_matrix(reconstructed_a, dataArgs["n_max"], dataArgs[
                "diag_value"])  # pad adjacency matrix to allow less nodes than n_max and fill diagonal

            ## 1) create adjacency plot_____________________________

            figure[0:n, j * n: (j + 1) * n] = reconstructed_a

            ## 2) create graph plot_____________________________

            # reconstruct graph
            reconstructed_a = unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"])
            g = nx.from_numpy_matrix(reconstructed_a)

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
                x_decoded = param_decoder.predict(z_sample)

                g, y_pos, params, param_values = decode_param(analyzeArgs, dataArgs, scaler, x_decoded)

                ## convert graph to adjacency
                g, reconstructed_a = sort_adjacency(g)
                reconstructed_a = pad_matrix(reconstructed_a, dataArgs["n_max"], dataArgs[
                    "diag_value"])  # pad adjacency matrix to allow less nodes than n_max and fill diagonal

                ## 1) create adjacency plots_____________________________________

                figure[i * n: (i + 1) * n, j * n: (j + 1) * n] = reconstructed_a

                ## 2) create graph plot_____________________________

                ## reconstruct graph
                reconstructed_a = unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"])
                g = nx.from_numpy_matrix(reconstructed_a)

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

        ## fill unobserved dimensions with mean of latent variable dimension
        for dim in range(0, len(z_sample[0])):
            z_sample[0][dim] = np.mean(z_mean[:, dim])

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

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample[0][analyzeArgs["z"][0]] = xi ** analyzeArgs["act_scale"]
                z_sample[0][analyzeArgs["z"][1]] = xi ** analyzeArgs["act_scale"]
                x_decoded = param_decoder.predict(z_sample)

                g, y_pos, params, param_values = decode_param(analyzeArgs, dataArgs, scaler, x_decoded)

                ## convert graph to adjacency
                g, reconstructed_a = sort_adjacency(g)
                reconstructed_a = pad_matrix(reconstructed_a, dataArgs["n_max"], dataArgs[
                    "diag_value"])  # pad adjacency matrix to allow less nodes than n_max and fill diagonal
                ## 1) create adjacency plot_____________________________

                figure[i * n: (i + 1) * n,
                j * n: (j + 1) * n] = reconstructed_a

                ## 2) create graph plot_____________________________

                ## reconstruct graph
                reconstructed_a = unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"])
                g = nx.from_numpy_matrix(reconstructed_a)

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
        plt.xlabel("z_" + str(analyzeArgs["z"][0]), fontweight='bold')
        plt.ylabel("z_" + str(analyzeArgs["z"][1]), fontweight='bold')
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

        if analyzeArgs["save_plots"] == True:
            filename = os.path.join(model_name, "digits_over_latent.png")
            plt.savefig(filename)


def generate_param_topol_manifold(analyzeArgs, modelArgs, dataArgs, models, data, color_map, batch_size, scaler):
    print("latent dimensions:", modelArgs["latent_dim"])

    if modelArgs["param_loss"] == False:
        sys.exit("modelArgs[param_loss] should be True")
    else:
        encoder, graph_decoder, param_decoder = models  # trained models

    x_test, y_test = data

    # display a 2D plot of the digit classes in the latent space
    z_mean, z_log_var, z = encoder.predict(x_test, batch_size)

    ## Latent Space Dimension is 1 ______________________

    if modelArgs["latent_dim"] == 1:

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

        ## 1) create graph topol plots_______________________________________________

        fig, axs = plt.subplots(1, analyzeArgs["size_of_manifold"], figsize=(10, 10))
        # fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()

        for j, xi in enumerate(grid_x):
            z_sample[0][0] = xi ** analyzeArgs["act_scale"]
            x_decoded = param_decoder.predict(z_sample)

            g, y_pos, params, param_values = decode_param(analyzeArgs, dataArgs, scaler, x_decoded)

            # compute index for the subplot, and set this subplot as current
            plt.sca(axs[j])

            ## create the plot_____________________________________________

            colors = ["midnightblue", "steelblue", "skyblue"]

            plt.bar(y_pos, param_values, color=colors, align='center')
            plt.plot([-1, 2], [0.25, 0.25], color='grey', linestyle='dashed')
            plt.plot([-1, 2], [0.5, 0.5], color='grey', linestyle='dashed')
            plt.plot([-1, 2], [0.75, 0.75], color='grey', linestyle='dashed')
            plt.xticks(y_pos, params)

            axs[xi].set_axis_off()

        # import matplotlib.patches as mpatches

        # density_patch = mpatches.Patch(color='midnightblue', label='density')
        # cluster_patch = mpatches.Patch(color='blue', label='cluster_coef')
        # assort_patch = mpatches.Patch(color='steelblue', label='assort')
        # avg_degree_patch = mpatches.Patch(color='skyblue', label='avg_degree')
        # axs[-1].legend(handles=[density_patch, cluster_patch, assort_patch, avg_degree_patch])

        if analyzeArgs["save_plots"] == True:
            filename = os.path.join(model_name, "digits_over_latent.png")
            plt.savefig(filename)

    if modelArgs["latent_dim"] == 2:

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

        ## 1) create graph topol plots_______________________________________________

        fig, axs = plt.subplots(analyzeArgs["size_of_manifold"], analyzeArgs["size_of_manifold"], figsize=(8, 8))
        # fig.subplots_adjust(hspace = .5, wspace=.001)

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                xi_value = xi ** analyzeArgs["act_scale"]
                yi_value = yi ** analyzeArgs["act_scale"]

                z_sample = np.array([[xi_value, yi_value]])
                x_decoded = param_decoder.predict(z_sample)

                g, y_pos, params, param_values = decode_param(analyzeArgs, dataArgs, scaler, x_decoded)

                # compute index for the subplot, and set this subplot as current
                plt.sca(axs[i, j])

                ## create the plot_____________________________________________

                colors = ["midnightblue", "steelblue", "skyblue"]

                plt.bar(y_pos, param_values, color=colors, align='center')
                plt.plot([-1, 2], [0.25, 0.25], color='grey', linestyle='dashed')
                plt.plot([-1, 2], [0.5, 0.5], color='grey', linestyle='dashed')
                plt.plot([-1, 2], [0.75, 0.75], color='grey', linestyle='dashed')
                plt.xticks(y_pos, params)

                axs[i, j].set_axis_off()

        if analyzeArgs["save_plots"] == True:
            filename = os.path.join(model_name, "digits_over_latent.png")
            plt.savefig(filename)

    if modelArgs["latent_dim"] > 2:

        z_sample = np.zeros(modelArgs["latent_dim"])
        z_sample = np.reshape(z_sample, (1, modelArgs["latent_dim"]))

        ## fill unobserved dimensions with mean of latent variable dimension
        for dim in range(0, len(z_sample[0])):
            z_sample[0][dim] = np.mean(z_mean[:, dim])

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

        ## 1) create graph topol plots_______________________________________________

        fig, axs = plt.subplots(analyzeArgs["size_of_manifold"], analyzeArgs["size_of_manifold"], figsize=(10, 10))
        # fig.subplots_adjust(hspace = .5, wspace=.001)

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample[0][analyzeArgs["z"][0]] = xi ** analyzeArgs["act_scale"]
                z_sample[0][analyzeArgs["z"][1]] = xi ** analyzeArgs["act_scale"]
                x_decoded = param_decoder.predict(z_sample)

                g, y_pos, params, param_values = decode_param(analyzeArgs, dataArgs, scaler, x_decoded)

                # compute index for the subplot, and set this subplot as current
                plt.sca(axs[i, j])

                ## create the plot_____________________________________________

                colors = ["midnightblue", "steelblue", "skyblue"]

                plt.bar(y_pos, param_values, color=colors, align='center')
                plt.plot([-1, 2], [0.25, 0.25], color='grey', linestyle='dashed')
                plt.plot([-1, 2], [0.5, 0.5], color='grey', linestyle='dashed')
                plt.plot([-1, 2], [0.75, 0.75], color='grey', linestyle='dashed')
                plt.xticks(y_pos, params)

                axs[i, j].set_axis_off()

        if analyzeArgs["save_plots"] == True:
            filename = os.path.join(model_name, "digits_over_latent.png")
            plt.savefig(filename)



