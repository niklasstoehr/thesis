from support.preprocessing import reconstruct_adjacency, unpad_matrix, unpad_attr, pad_attr

from support.plotting import shiftedColorMap
import networkx as nx
import numpy as np
import seaborn as sns

from matplotlib import pylab as plt
import os




def generate_single_features(analyzeArgs, modelArgs, dataArgs, models, orig_cmap):
    encoder, decoder = models  # trained models

    print("latent dimensions:", modelArgs["latent_dim"])

    z_sample = np.zeros(modelArgs["latent_dim"])
    z_sample = np.reshape(z_sample, (1, modelArgs["latent_dim"]))

    for i, dim in enumerate(analyzeArgs["z"]):
        z_sample[0][dim] = analyzeArgs["activations"][i]

    [f_decoded, a_decoded] = decoder.predict(z_sample)
    a_decoded = np.squeeze(a_decoded[0])
    f_decoded = f_decoded[0]

    ## reconstruct upper triangular adjacency matrix
    reconstructed_a = reconstruct_adjacency(a_decoded, dataArgs["clip"], dataArgs["diag_offset"])
    reconstructed_a, nodes_n = unpad_matrix(reconstructed_a, dataArgs["diag_value"], 0.1, dataArgs["fix_n"])

    reconstructed_f = unpad_attr(f_decoded, nodes_n, analyzeArgs, dataArgs)

    print("nodes_n:", nodes_n)
    print("node attributes:", reconstructed_f)

    ## reconstruct graph
    g = nx.from_numpy_matrix(reconstructed_a)
    if reconstructed_f.shape[0] > 0:
        fixed_cmap = shiftedColorMap(orig_cmap, start=min(reconstructed_f), midpoint=0.5, stop=max(reconstructed_f),
                                     name='fixed')
    else:
        fixed_cmap = shiftedColorMap(orig_cmap, start=0.5, midpoint=0.5, stop=0.5, name='fixed')
    nx.draw(g, node_color=reconstructed_f, font_color='white', cmap=fixed_cmap)
    plt.show()

    ax = sns.distplot(reconstructed_f, rug=True)
    ax.set_title('Node Attribute Distribution', fontweight="bold")
    ax.set(xlabel="node attributes", ylabel="frequency")
    plt.show()







## DECODER - Latent Space Interpolation____________________________

def generate_manifold_features(analyzeArgs, modelArgs, dataArgs, models, data, orig_cmap, batch_size=128):
    print("latent dimensions:", modelArgs["latent_dim"])
    encoder, decoder = models  # trained models

    F_org, [A_fil, A] = data
    z_mean, z_log_var, _ = encoder.predict([F_org, A_fil], batch_size=batch_size)

    ## Latent Space Dimension is 2 ______________________

    if modelArgs["latent_dim"] == 2:

        ## 1) create adjacency plots_______________________________________________

        # display a 30x30 2D manifold of digits
        n = dataArgs["n_max"]  # number of nodes
        figure = np.zeros((analyzeArgs["size_of_manifold"] * n, analyzeArgs["size_of_manifold"] * n, 3))

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

                [f_decoded, a_decoded] = decoder.predict(z_sample)
                a_decoded = np.squeeze(a_decoded[0])
                f_decoded = f_decoded[0]

                ## reconstruct upper triangular adjacency matrix
                reconstructed_a_padded = reconstruct_adjacency(a_decoded, dataArgs["clip"], dataArgs["diag_offset"])

                reconstructed_a, nodes_n = unpad_matrix(reconstructed_a_padded, dataArgs["diag_value"], 0.1,
                                                        dataArgs["fix_n"])
                reconstructed_f = unpad_attr(f_decoded, nodes_n, analyzeArgs, dataArgs)

                ## build fixed cmap
                if reconstructed_f.shape[0] > 0:
                    fixed_cmap = shiftedColorMap(orig_cmap, start=min(reconstructed_f), midpoint=0.5,
                                                 stop=max(reconstructed_f), name='fixed')
                else:
                    fixed_cmap = shiftedColorMap(orig_cmap, start=0.5, midpoint=0.5, stop=0.5, name='fixed')

                ## adjust colour reconstructed_a_padded according to features
                feature_a = np.copy(reconstructed_a_padded)
                feature_a = np.tile(feature_a[:, :, None], [1, 1, 3])  ## broadcast 1 channel to 3

                for node in range(0, nodes_n):
                    color = fixed_cmap(reconstructed_f[node])[:3]
                    feature_a[node, :node + 1] = feature_a[node, :node + 1] * color
                    feature_a[:node, node] = feature_a[:node, node] * color

                ## 1) create adjacency plots_____________________________________

                figure[i * n: (i + 1) * n, j * n: (j + 1) * n, :] = feature_a

                ## 2) create graph plot_____________________________

                ## reconstruct graph
                g = nx.from_numpy_matrix(reconstructed_a)

                # compute index for the subplot, and set this subplot as current
                plt.sca(axs[i, j])
                nx.draw(g, node_size=12, node_color=reconstructed_f, width=0.2, font_color='white', cmap=fixed_cmap)
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
        figure = np.zeros((analyzeArgs["size_of_manifold"] * n, analyzeArgs["size_of_manifold"] * n, 3))

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

                [f_decoded, a_decoded] = decoder.predict(z_sample)
                a_decoded = np.squeeze(a_decoded[0])
                f_decoded = f_decoded[0]

                ## reconstruct upper triangular adjacency matrix
                reconstructed_a_padded = reconstruct_adjacency(a_decoded, dataArgs["clip"], dataArgs["diag_offset"])

                reconstructed_a, nodes_n = unpad_matrix(reconstructed_a_padded, dataArgs["diag_value"], 0.1,
                                                        dataArgs["fix_n"])
                reconstructed_f = unpad_attr(f_decoded, nodes_n, analyzeArgs, dataArgs)

                ## build fixed cmap
                if reconstructed_f.shape[0] > 0:
                    fixed_cmap = shiftedColorMap(orig_cmap, start=min(reconstructed_f), midpoint=0.5,
                                                 stop=max(reconstructed_f), name='fixed')
                else:
                    fixed_cmap = shiftedColorMap(orig_cmap, start=0.5, midpoint=0.5, stop=0.5, name='fixed')

                ## adjust colour reconstructed_a_padded according to features
                feature_a = np.copy(reconstructed_a_padded)
                feature_a = np.tile(feature_a[:, :, None], [1, 1, 3])  ## broadcast 1 channel to 3

                for node in range(0, nodes_n):
                    color = fixed_cmap(reconstructed_f[node])[:3]
                    feature_a[node, :node + 1] = feature_a[node, :node + 1] * color
                    feature_a[:node, node] = feature_a[:node, node] * color

                ## 1) create adjacency plots_____________________________________

                figure[i * n: (i + 1) * n, j * n: (j + 1) * n, :] = feature_a

                ## 2) create graph plot_____________________________

                ## reconstruct graph
                g = nx.from_numpy_matrix(reconstructed_a)

                # compute index for the subplot, and set this subplot as current
                plt.sca(axs[i, j])
                nx.draw(g, node_size=12, node_color=reconstructed_f, width=0.2, font_color='white', cmap=fixed_cmap)
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















