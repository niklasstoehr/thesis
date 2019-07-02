from support.preprocessing import reconstruct_adjacency, unpad_matrix
from sklearn.metrics import precision_recall_fscore_support
import networkx as nx
import numpy as np
from matplotlib import pylab as plt
import os


## DECODER - Latent Space Interpolation____________________________

def compare_manifold_adjacency(g_original, a_original, analyzeArgs, modelArgs, dataArgs, models, data, color_map, batch_size=128):

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
        figure = np.zeros((1 * n, analyzeArgs["size_of_manifold"] * n, 3))

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

            comparison_matrix = np.zeros((n, n, 3))
            reconstructed_a = reconstruct_adjacency(x_decoded, dataArgs["clip"], dataArgs["diag_offset"])

            # reconstruct graph
            g = nx.from_numpy_matrix(unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"]))

            # metrics per graph
            metrics = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

            for x in range(0, n):
                for y in range(0, n):

                    if a_original[x, y] == reconstructed_a[x, y] and a_original[x, y] == 1:  # correct
                        comparison_matrix[x, y, :] = [0, 90, 0]  # green
                        if x != y:
                            metrics["tp"] = metrics.get("tp") + 1

                    elif a_original[x, y] == reconstructed_a[x, y] and a_original[x, y] == 0:  # correct
                        comparison_matrix[x, y, :] = [255, 255, 255]  # black
                        if x != y:
                            metrics["tn"] = metrics.get("tn") + 1

                    elif a_original[x, y] != reconstructed_a[x, y] and a_original[x, y] == 1:  # underfit

                        if x < len(g) and y < len(g):
                            comparison_matrix[x, y, :] = [90, 0, 0]  # red   # missed
                            if x != y:
                                metrics["fn"] = metrics.get("fn") + 1
                        else:
                            comparison_matrix[x, y, :] = [150, 150, 150]  # grey   # not possible since too small

                    elif a_original[x, y] != reconstructed_a[x, y] and a_original[x, y] == 0:  # overfit
                        comparison_matrix[x, y, :] = [70, 70, 0]  # yellow
                        if x != y:
                            metrics["fp"] = metrics.get("fp") + 1

            ## 1) create adjacency plots_____________________________________

            figure[0:n, j * n: (j + 1) * n, :] = comparison_matrix

            ## 2) create metric plots _______________________________________________

            acc = (metrics["tp"] + metrics["tn"]) / (metrics["tn"] + metrics["fn"] + metrics["tp"] + metrics["fp"])

            if (metrics["tp"] + metrics["fp"]) > 0:
                prec = (metrics["tp"]) / (metrics["tp"] + metrics["fp"])
            else:
                prec = 0

            if (metrics["tp"] + metrics["fn"]) > 0:
                recall = (metrics["tp"]) / (metrics["tp"] + metrics["fn"])
            else:
                recall = 0

            if (recall + prec) > 0:
                f1 = 2 * (recall * prec) / (recall + prec)
            else:
                f1 = 0

            y_pos = np.arange(3)
            final_metrics = [prec, recall, f1]
            colors = ["skyblue", "skyblue", "midnightblue"]

            jx = np.unravel_index(j, axs.shape)
            plt.sca(axs[jx])

            plt.bar(y_pos, final_metrics, color=colors, align='center')
            plt.plot([-1, 3], [0.25, 0.25], color='grey', linestyle='dashed')
            plt.plot([-1, 3], [0.5, 0.5], color='grey', linestyle='dashed')
            plt.plot([-1, 3], [0.75, 0.75], color='grey', linestyle='dashed')

            axs[jx].set_axis_off()
            axs[jx].set(ylabel='z_0')

        start_range = n // 2
        end_range = (analyzeArgs["size_of_manifold"] - 1) * n + start_range + 1
        pixel_range = np.arange(start_range, end_range, n)
        sample_range_x = np.round(grid_x, 1)

        # Plot_____________________________

        plt.figure(figsize=(10, 10))
        plt.xticks(pixel_range, sample_range_x)
        plt.xlabel("z_0", fontweight='bold')
        plt.imshow((figure * 255).astype(np.uint8))
        plt.show()

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

        ## 2) create metric plots _______________________________________________

        fig, axs = plt.subplots(analyzeArgs["size_of_manifold"], analyzeArgs["size_of_manifold"], figsize=(10, 10))

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):

                xi_value = xi ** analyzeArgs["act_scale"]
                yi_value = yi ** analyzeArgs["act_scale"]

                z_sample = np.array([[xi_value, yi_value]])
                x_decoded = decoder.predict(z_sample)

                comparison_matrix = np.zeros((n, n, 3))
                reconstructed_a = reconstruct_adjacency(x_decoded, dataArgs["clip"], dataArgs["diag_offset"])

                # reconstruct graph
                g = nx.from_numpy_matrix(unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"]))

                # metrics per graph
                metrics = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

                for x in range(0, n):
                    for y in range(0, n):

                        if a_original[x, y] == reconstructed_a[x, y] and a_original[x, y] == 1:  # correct
                            comparison_matrix[x, y, :] = [0, 90, 0]  # green
                            if x != y:
                                metrics["tp"] = metrics.get("tp") + 1

                        elif a_original[x, y] == reconstructed_a[x, y] and a_original[x, y] == 0:  # correct
                            comparison_matrix[x, y, :] = [255, 255, 255]  # black
                            if x != y:
                                metrics["tn"] = metrics.get("tn") + 1

                        elif a_original[x, y] != reconstructed_a[x, y] and a_original[x, y] == 1:  # underfit

                            if x < len(g) and y < len(g):
                                comparison_matrix[x, y, :] = [90, 0, 0]  # red   # missed
                                if x != y:
                                    metrics["fn"] = metrics.get("fn") + 1
                            else:
                                comparison_matrix[x, y, :] = [150, 150, 150]  # grey   # not possible since too small

                        elif a_original[x, y] != reconstructed_a[x, y] and a_original[x, y] == 0:  # overfit
                            comparison_matrix[x, y, :] = [70, 70, 0]  # yellow
                            if x != y:
                                metrics["fp"] = metrics.get("fp") + 1

                ## 1) create adjacency plots_____________________________________

                figure[i * n: (i + 1) * n, j * n: (j + 1) * n, :] = comparison_matrix

                ## 2) create metric plots _______________________________________________

                acc = (metrics["tp"] + metrics["tn"]) / (metrics["tn"] + metrics["fn"] + metrics["tp"] + metrics["fp"])

                if (metrics["tp"] + metrics["fp"]) > 0:
                    prec = (metrics["tp"]) / (metrics["tp"] + metrics["fp"])
                else:
                    prec = 0

                if (metrics["tp"] + metrics["fn"]) > 0:
                    recall = (metrics["tp"]) / (metrics["tp"] + metrics["fn"])
                else:
                    recall = 0

                if (recall + prec) > 0:
                    f1 = 2 * (recall * prec) / (recall + prec)
                else:
                    f1 = 0

                y_pos = np.arange(3)
                final_metrics = [prec, recall, f1]
                colors = ["skyblue", "skyblue", "midnightblue"]

                plt.sca(axs[i, j])
                plt.bar(y_pos, final_metrics, color=colors, align='center')
                plt.plot([-1, 3], [0.25, 0.25], color='grey', linestyle='dashed')
                plt.plot([-1, 3], [0.5, 0.5], color='grey', linestyle='dashed')
                plt.plot([-1, 3], [0.75, 0.75], color='grey', linestyle='dashed')
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
        plt.imshow((figure * 255).astype(np.uint8))
        plt.show()

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

        ## 2) create metric plots _______________________________________________

        fig, axs = plt.subplots(analyzeArgs["size_of_manifold"], analyzeArgs["size_of_manifold"], figsize=(10, 10))

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):

                z_sample[0][analyzeArgs["z"][0]] = xi ** analyzeArgs["act_scale"]
                z_sample[0][analyzeArgs["z"][1]] = xi ** analyzeArgs["act_scale"]
                x_decoded = decoder.predict(z_sample)

                comparison_matrix = np.zeros((n, n, 3))
                reconstructed_a = reconstruct_adjacency(x_decoded, dataArgs["clip"], dataArgs["diag_offset"])

                # reconstruct graph
                g = nx.from_numpy_matrix(unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"]))

                # metrics per graph
                metrics = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

                for x in range(0, n):
                    for y in range(0, n):

                        if a_original[x, y] == reconstructed_a[x, y] and a_original[x, y] == 1:  # correct
                            comparison_matrix[x, y, :] = [0, 90, 0]  # green
                            if x != y:
                                metrics["tp"] = metrics.get("tp") + 1

                        elif a_original[x, y] == reconstructed_a[x, y] and a_original[x, y] == 0:  # correct
                            comparison_matrix[x, y, :] = [255, 255, 255]  # black
                            if x != y:
                                metrics["tn"] = metrics.get("tn") + 1

                        elif a_original[x, y] != reconstructed_a[x, y] and a_original[x, y] == 1:  # underfit

                            if x < len(g) and y < len(g):
                                comparison_matrix[x, y, :] = [90, 0, 0]  # red   # missed
                                if x != y:
                                    metrics["fn"] = metrics.get("fn") + 1
                            else:
                                comparison_matrix[x, y, :] = [150, 150, 150]  # grey   # not possible since too small

                        elif a_original[x, y] != reconstructed_a[x, y] and a_original[x, y] == 0:  # overfit
                            comparison_matrix[x, y, :] = [70, 70, 0]  # yellow
                            if x != y:
                                metrics["fp"] = metrics.get("fp") + 1

                ## 1) create adjacency plots_____________________________________

                figure[i * n: (i + 1) * n, j * n: (j + 1) * n, :] = comparison_matrix

                ## 2) create metric plots _______________________________________________

                acc = (metrics["tp"] + metrics["tn"]) / (metrics["tn"] + metrics["fn"] + metrics["tp"] + metrics["fp"])

                if (metrics["tp"] + metrics["fp"]) > 0:
                    prec = (metrics["tp"]) / (metrics["tp"] + metrics["fp"])
                else:
                    prec = 0

                if (metrics["tp"] + metrics["fn"]) > 0:
                    recall = (metrics["tp"]) / (metrics["tp"] + metrics["fn"])
                else:
                    recall = 0

                if (recall + prec) > 0:
                    f1 = 2 * (recall * prec) / (recall + prec)
                else:
                    f1 = 0

                y_pos = np.arange(3)
                final_metrics = [prec, recall, f1]
                colors = ["skyblue", "skyblue", "midnightblue"]

                plt.sca(axs[i, j])
                plt.bar(y_pos, final_metrics, color=colors, align='center')
                plt.plot([-1, 3], [0.25, 0.25], color='grey', linestyle='dashed')
                plt.plot([-1, 3], [0.5, 0.5], color='grey', linestyle='dashed')
                plt.plot([-1, 3], [0.75, 0.75], color='grey', linestyle='dashed')
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
        plt.imshow((figure * 255).astype(np.uint8))
        plt.show()












def compare_topol_manifold(g_original, a_original, analyzeArgs, modelArgs, dataArgs, models, data, color_map,
                           batch_size=128):
    print("latent dimensions:", modelArgs["latent_dim"])

    encoder, decoder = models  # trained models
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

        degree_sequence_original = sorted([d for n, d in g_original.degree()], reverse=True)  # degree sequence

        for j, xi in enumerate(grid_x):

            z_sample[0][0] = xi ** analyzeArgs["act_scale"]
            x_decoded = decoder.predict(z_sample)

            ## reconstruct upper triangular adjacency matrix
            reconstructed_a = reconstruct_adjacency(x_decoded, dataArgs["clip"], dataArgs["diag_offset"])

            # reconstruct graph
            reconstructed_a = unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"])
            g = nx.from_numpy_matrix(reconstructed_a)

            ## Obtain Graph Topologies____________________________________

            density_original = nx.density(g_original)
            density = nx.density(g)

            diameter_original = nx.diameter(g_original)
            if len(g) > 0:
                if nx.is_connected(g):
                    diameter = nx.diameter(g)
            else:
                diameter = -1

            cluster_coef_original = nx.average_clustering(g_original)
            if len(g) > 0:
                cluster_coef = nx.average_clustering(g)
            else:
                cluster_coef = 0

            assort_original = nx.degree_assortativity_coefficient(g_original, x='out', y='in')
            if len(g) > 0:
                if g.number_of_edges() > 0:
                    assort = nx.degree_assortativity_coefficient(g, x='out', y='in')
            else:
                assort = 0

            edges_original = g_original.number_of_edges()
            if len(g) > 0:
                edges = g.number_of_edges()
            else:
                edges = 0

            avg_degree_original = sum(i for i in nx.degree_centrality(g_original).values()) / len(
                nx.degree_centrality(g_original).keys())
            if len(g) > 0:
                avg_degree = sum(i for i in nx.degree_centrality(g).values()) / len(nx.degree_centrality(g).keys())
            else:
                avg_degree = 0

            # compute index for the subplot, and set this subplot as current
            jx = np.unravel_index(j, axs.shape)
            plt.sca(axs[jx])

            ## create the plot_____________________________________________

            if analyzeArgs["plot"] == "topol":

                topol = ("cluster_coef", "assort", "avg_degree")
                colors = ["midnightblue", "steelblue", "skyblue"]

                y_pos = np.arange(len(topol))
                topol_values = [cluster_coef, assort, avg_degree]
                topol_values_original = [cluster_coef_original, assort_original, avg_degree_original]
                # plt.bar(y_pos, topol_values_original, color=colors, fill=False, align='center')
                plt.hlines(topol_values_original[0], -0.5, 0.5)
                plt.hlines(topol_values_original[1], 0.5, 1.5)
                plt.hlines(topol_values_original[2], 1.5, 2.5)
                plt.bar(y_pos, topol_values, color=colors, align='center')
                plt.xticks(y_pos, topol)


            elif analyzeArgs["plot"] == "topol_diff":

                topol = ("cluster_coef", "assort", "avg_degree")
                # pal = sns.color_palette("RdYlGn", len(topol))

                topol_values = [cluster_coef, assort, avg_degree]
                topol_values_original = [cluster_coef_original, assort_original, avg_degree_original]

                x_pos = np.arange(len(topol))
                topol_differences = (np.asarray(topol_values_original) - np.asarray(topol_values))

                # color = [(x/10.0, x/20.0, 0.75) for x in 10*(np.abs(topol_differences) / np.sum(np.abs(topol_differences)))] # <-- Quick gradient example along the Red/Green dimensions.

                from matplotlib import cm

                # colors = cm.YlOrRd(np.abs(topol_differences) / float(max(np.abs(topol_differences))))
                # colors = cm.Blues(np.abs(topol_differences) / float(max(np.abs(topol_differences))))
                colors = cm.RdYlGn_r(np.abs(topol_differences) / float(max(np.abs(topol_differences))))
                # colors = colors[::-1]
                plt.bar(range(len(topol_differences)), topol_differences, color=colors)


            elif analyzeArgs["plot"] == "distr":

                degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence

                degree_sequence = np.asarray(degree_sequence) / sum(degree_sequence)  # normalize degree sequence
                degree_sequence = np.repeat(degree_sequence, (len(degree_sequence_original) / len(
                    degree_sequence)))  # stretch normalize degree sequence to match length
                degree_sequence_original = np.asarray(degree_sequence_original) / sum(
                    degree_sequence_original)  # normalize original degree sequence

                plt.plot(degree_sequence_original, color="midnightblue", linestyle='dashed')
                plt.plot(degree_sequence, color="steelblue")

            axs[jx].set_axis_off()

        # import matplotlib.patches as mpatches

        # density_patch = mpatches.Patch(color='midnightblue', label='density')
        # cluster_patch = mpatches.Patch(color='blue', label='cluster_coef')
        # assort_patch = mpatches.Patch(color='steelblue', label='assort')
        # avg_degree_patch = mpatches.Patch(color='skyblue', label='avg_degree')
        # axs[-1].legend(handles=[density_patch, cluster_patch, assort_patch, avg_degree_patch])

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

        degree_sequence_original = sorted([d for n, d in g_original.degree()], reverse=True)  # degree sequence

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):

                xi_value = xi ** analyzeArgs["act_scale"]
                yi_value = yi ** analyzeArgs["act_scale"]

                z_sample = np.array([[xi_value, yi_value]])
                x_decoded = decoder.predict(z_sample)

                ## reconstruct upper triangular adjacency matrix
                reconstructed_a = reconstruct_adjacency(x_decoded, dataArgs["clip"], dataArgs["diag_offset"])

                ## reconstruct graph
                reconstructed_a = unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"])
                g = nx.from_numpy_matrix(reconstructed_a)

                ## Obtain Graph Topologies____________________________________

                density_original = nx.density(g_original)
                density = nx.density(g)

                diameter_original = nx.diameter(g_original)
                if len(g) > 0:
                    if nx.is_connected(g):
                        diameter = nx.diameter(g)
                else:
                    diameter = -1

                cluster_coef_original = nx.average_clustering(g_original)
                if len(g) > 0:
                    cluster_coef = nx.average_clustering(g)
                else:
                    cluster_coef = 0

                assort_original = nx.degree_assortativity_coefficient(g_original, x='out', y='in')
                if len(g) > 0:
                    if g.number_of_edges() > 0:
                        assort = nx.degree_assortativity_coefficient(g, x='out', y='in')
                else:
                    assort = 0

                edges_original = g_original.number_of_edges()
                if len(g) > 0:
                    edges = g.number_of_edges()
                else:
                    edges = 0

                avg_degree_original = sum(i for i in nx.degree_centrality(g_original).values()) / len(
                    nx.degree_centrality(g_original).keys())
                if len(g) > 0:
                    avg_degree = sum(i for i in nx.degree_centrality(g).values()) / len(nx.degree_centrality(g).keys())
                else:
                    avg_degree = 0

                # compute index for the subplot, and set this subplot as current
                plt.sca(axs[i, j])

                ## create the plot_____________________________________________

                if analyzeArgs["plot"] == "topol":

                    topol = ("cluster_coef", "assort", "avg_degree")
                    colors = ["midnightblue", "steelblue", "skyblue"]

                    y_pos = np.arange(len(topol))
                    topol_values = [cluster_coef, assort, avg_degree]
                    topol_values_original = [cluster_coef_original, assort_original, avg_degree_original]
                    # plt.bar(y_pos, topol_values_original, color=colors, fill=False, align='center')
                    plt.hlines(topol_values_original[0], -0.5, 0.5)
                    plt.hlines(topol_values_original[1], 0.5, 1.5)
                    plt.hlines(topol_values_original[2], 1.5, 2.5)
                    plt.bar(y_pos, topol_values, color=colors, align='center')
                    plt.xticks(y_pos, topol)


                elif analyzeArgs["plot"] == "topol_diff":

                    topol = ("cluster_coef", "assort", "avg_degree")
                    # pal = sns.color_palette("RdYlGn", len(topol))

                    topol_values = [cluster_coef, assort, avg_degree]
                    topol_values_original = [cluster_coef_original, assort_original, avg_degree_original]

                    x_pos = np.arange(len(topol))
                    topol_differences = (np.asarray(topol_values_original) - np.asarray(topol_values))

                    # color = [(x/10.0, x/20.0, 0.75) for x in 10*(np.abs(topol_differences) / np.sum(np.abs(topol_differences)))] # <-- Quick gradient example along the Red/Green dimensions.

                    from matplotlib import cm

                    # colors = cm.YlOrRd(np.abs(topol_differences) / float(max(np.abs(topol_differences))))
                    # colors = cm.Blues(np.abs(topol_differences) / float(max(np.abs(topol_differences))))
                    colors = cm.RdYlGn_r(np.abs(topol_differences) / float(max(np.abs(topol_differences))))
                    # colors = colors[::-1]
                    plt.bar(range(len(topol_differences)), topol_differences, color=colors)


                elif analyzeArgs["plot"] == "distr":

                    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence

                    degree_sequence = np.asarray(degree_sequence) / sum(degree_sequence)  # normalize degree sequence
                    degree_sequence = np.repeat(degree_sequence, (len(degree_sequence_original) / len(
                        degree_sequence)))  # stretch normalize degree sequence to match length
                    degree_sequence_original = np.asarray(degree_sequence_original) / sum(
                        degree_sequence_original)  # normalize original degree sequence

                    plt.plot(degree_sequence_original, color="midnightblue", linestyle='dashed')
                    plt.plot(degree_sequence, color="steelblue")

                axs[i, j].set_axis_off()

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

        degree_sequence_original = sorted([d for n, d in g_original.degree()], reverse=True)  # degree sequence

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):

                z_sample[0][analyzeArgs["z"][0]] = xi ** analyzeArgs["act_scale"]
                z_sample[0][analyzeArgs["z"][1]] = xi ** analyzeArgs["act_scale"]
                x_decoded = decoder.predict(z_sample)

                ## reconstruct upper triangular adjacency matrix
                reconstructed_a = reconstruct_adjacency(x_decoded, dataArgs["clip"], dataArgs["diag_offset"])

                ## reconstruct graph
                reconstructed_a = unpad_matrix(reconstructed_a, dataArgs["diag_value"], dataArgs["fix_n"])
                g = nx.from_numpy_matrix(reconstructed_a)

                ## Obtain Graph Topologies____________________________________

                density_original = nx.density(g_original)
                density = nx.density(g)

                diameter_original = nx.diameter(g_original)
                if len(g) > 0:
                    if nx.is_connected(g):
                        diameter = nx.diameter(g)
                else:
                    diameter = -1

                cluster_coef_original = nx.average_clustering(g_original)
                if len(g) > 0:
                    cluster_coef = nx.average_clustering(g)
                else:
                    cluster_coef = 0

                assort_original = nx.degree_assortativity_coefficient(g_original, x='out', y='in')
                if len(g) > 0:
                    if g.number_of_edges() > 0:
                        assort = nx.degree_assortativity_coefficient(g, x='out', y='in')
                else:
                    assort = 0

                edges_original = g_original.number_of_edges()
                if len(g) > 0:
                    edges = g.number_of_edges()
                else:
                    edges = 0

                avg_degree_original = sum(i for i in nx.degree_centrality(g_original).values()) / len(
                    nx.degree_centrality(g_original).keys())
                if len(g) > 0:
                    avg_degree = sum(i for i in nx.degree_centrality(g).values()) / len(nx.degree_centrality(g).keys())
                else:
                    avg_degree = 0

                # compute index for the subplot, and set this subplot as current
                plt.sca(axs[i, j])

                ## create the plot_____________________________________________

                ## create the plot_____________________________________________

                if analyzeArgs["plot"] == "topol":

                    topol = ("cluster_coef", "assort", "avg_degree")
                    colors = ["midnightblue", "steelblue", "skyblue"]

                    y_pos = np.arange(len(topol))
                    topol_values = [cluster_coef, assort, avg_degree]
                    topol_values_original = [cluster_coef_original, assort_original, avg_degree_original]
                    # plt.bar(y_pos, topol_values_original, color=colors, fill=False, align='center')
                    plt.hlines(topol_values_original[0], -0.5, 0.5)
                    plt.hlines(topol_values_original[1], 0.5, 1.5)
                    plt.hlines(topol_values_original[2], 1.5, 2.5)
                    plt.bar(y_pos, topol_values, color=colors, align='center')
                    plt.xticks(y_pos, topol)

                elif analyzeArgs["plot"] == "topol_diff":

                    topol = ("cluster_coef", "assort", "avg_degree")
                    # pal = sns.color_palette("RdYlGn", len(topol))

                    topol_values = [cluster_coef, assort, avg_degree]
                    topol_values_original = [cluster_coef_original, assort_original, avg_degree_original]

                    x_pos = np.arange(len(topol))
                    topol_differences = (np.asarray(topol_values_original) - np.asarray(topol_values))

                    # color = [(x/10.0, x/20.0, 0.75) for x in 10*(np.abs(topol_differences) / np.sum(np.abs(topol_differences)))] # <-- Quick gradient example along the Red/Green dimensions.

                    from matplotlib import cm

                    # colors = cm.YlOrRd(np.abs(topol_differences) / float(max(np.abs(topol_differences))))
                    # colors = cm.Blues(np.abs(topol_differences) / float(max(np.abs(topol_differences))))
                    colors = cm.RdYlGn_r(np.abs(topol_differences) / float(max(np.abs(topol_differences))))
                    # colors = colors[::-1]
                    plt.bar(range(len(topol_differences)), topol_differences, color=colors)

                elif analyzeArgs["plot"] == "distr":

                    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence

                    degree_sequence = np.asarray(degree_sequence) / sum(degree_sequence)  # normalize degree sequence
                    degree_sequence = np.repeat(degree_sequence, (len(degree_sequence_original) / len(
                        degree_sequence)))  # stretch normalize degree sequence to match length
                    degree_sequence_original = np.asarray(degree_sequence_original) / sum(
                        degree_sequence_original)  # normalize original degree sequence

                    plt.plot(degree_sequence_original, color="midnightblue", linestyle='dashed')
                    plt.plot(degree_sequence, color="steelblue")

                axs[i, j].set_axis_off()

