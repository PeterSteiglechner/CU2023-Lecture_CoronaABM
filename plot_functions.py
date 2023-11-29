# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import networkx as nx

plt.rcParams.update({"font.size": 10})



def plot_net(g, agents, filename=None):
    """
    Plot the network of g.
    Each risk group has a certain color
    :param g: a network graph consisting of indices of all the agents (stored in the list agents), links between them.
    :param agents: list of all agents
    :return:
    """
    # From https://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3
    # An excellent source if you want to define well-fitting, visible and commonly used colors/-schemes.
    # I choose
    #   Number=3,
    #   qualitative nature (maximum difference between the colors)
    colors_nodes = [
        (228 / 255, 26 / 255, 28 / 255),  # Red = RiskGroup 0
        (55 / 255, 126 / 255, 184 / 255),  # Blue = RiskGroup 1
        (77 / 255, 175 / 255, 74 / 255)  # Green = RiskGroup 2
    ]
    # From the colors, define colormap:
    cmap_nodes = mpl.colors.ListedColormap(colors_nodes)

    # For each node in the graph or agent in the list, add the group in the node_color array
    # Alternative:
    #   You can also plot the different infection states in a colormap of 9 colors.
    #   In this case you would simply add the state instead of the group as an attribute to the node
    #   and then plot these attributes
    node_colors = []
    for ag, n in zip(agents, g.nodes):
        node_colors.append(ag.group)
        # g.nodes[n]["group"] = ag.group
    # node_colors = [g.nodes.data("group")[n] for n in g.nodes]

    # Determine a good layout for the nodes (clustering close links)
    pos = nx.spring_layout(g, k=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    nx.draw(g, pos, ax=ax, node_color=node_colors, cmap=cmap_nodes, node_size=100)
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()

    # Get the node-degrees for each node (how many links does each node have)
    total_node_degrees = nx.adjacency_matrix(g).sum(axis=1)
    print("An agent has on average : {:.2f}+-{:.2f} links in the net".format(total_node_degrees.mean(),
                                                                             total_node_degrees.std()))
    return


def plot_states_network(agents, seed, t_array, network, k_ws, p_ws):
    fig = plt.figure(figsize=(16/2.54,9/2.54))
    ax = plt.axes()
    pos = nx.spring_layout(network)
    state_map={"susceptible":1, "latent":2, "infectious_preSymptom":3, "infectious_asymptomatic":4, "infectious_symptomatic":5, "recovered":6, "dead":7}
    states = [state_map[ag.health_state] for ag in agents]
    colorsList = ['grey', 'lime', 'orange', 'yellow', 'red', 'green', 'black']
    statesCmap = mpl.colors.ListedColormap(colorsList)
    bounds = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=7)

    pathcollection = nx.draw_networkx_nodes(network, pos, node_size=8, node_color=states, cmap=statesCmap, vmin=0.5, vmax=7.5, alpha=0.7)
    cbar = plt.colorbar(pathcollection, ticks=np.arange(1,8),norm=norm)
    cbar.ax.set_yticklabels(state_map.keys(), fontsize=10)
    nx.draw_networkx_edges(network, pos, edge_color="grey", alpha=0.4, width=0.5)
    ax.set_title(f"Network ({k_ws}, {p_ws}) at time t={t_array[-1]}, seed {seed}" , fontsize=10)
    fig.tight_layout()
    fname = f"figures/network_({k_ws},{p_ws})_t{t_array[-1]}_seed{seed}.png"
    plt.savefig(fname, dpi=600)
    return fname



def plot_statistics(time_array, results, states, title="", filename=None, ymax=140):
    """
    Plots the aggregate results over time
    :param results: Array with dim (len(t_array), len(states)+1):
            Column 0: time
            Column 1-9: aggregate numbers in this state at the corresponding time
    :param states: ["s", "l", "i_a", "i_ps", "i_s", "r", "d"]
    :param title: (string) The title added to the axis of the figure.
    :param filename: (string) Name of saved pdf figure in the folder specified by FOLDER
    :return:
    """
    fig = plt.figure(figsize=(16/2.54, 9/2.54))
    ax = fig.add_subplot(111)
    ax.set_xlim(time_array[0], time_array[-1])
    colorsList = ['grey', 'lime', 'orange', 'yellow', 'red', 'green', 'black']
    
    ax.plot(time_array, results[:, 0], "--", lw=1, label="susceptible", color=colorsList[0])
    ax.set_ylim(0, 1000)
    #plt.legend(loc="center left", fontsize=10)
    ax2 = ax.twinx()
    ax2.set_ylim(0, ymax)
    for n, state in enumerate(states[1:]):
        ax2.plot(time_array, results[:, n+1], lw=1, label=state, color=colorsList[n + 1])
    plt.legend(loc="center right")
    ax2.set_ylabel("# people in a specific state", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time [days]", fontsize=10)
    ax.set_ylabel("susceptible", fontsize=10)
    if filename is not None:
        fig.tight_layout()
        plt.savefig(filename + ".png", bbox_inches='tight', dpi=600)
    return






"""
AVANCED 2:
PLOT R VALUES OVER TIME
"""
def plot_r_values(_agents, n_convolve=20, fname=None):
    """ Plot all r values over time, via convolution """
    all_rs = np.array([ag.r for ag in _agents])
    all_tes = np.array([ag.t_e for ag in _agents])
    all_rs_without_nan = all_rs[np.where(np.isnan(all_rs) == False)]
    all_tes_without_nan = all_tes[np.where(np.isnan(all_tes) == False)]
    inds = np.argsort(all_tes_without_nan)
    rs_sort = all_rs_without_nan[inds]
    rs_conv = np.convolve(rs_sort, np.ones(n_convolve, ) / n_convolve, mode='valid')
    times_conv = np.convolve(all_tes_without_nan[inds], np.ones(n_convolve, ) / n_convolve, mode="valid")

    fig = plt.figure(figsize=(16/2.54, 9/2.54))
    ax = fig.add_subplot(111)
    ax.plot(times_conv, rs_conv)
    ax.set_title("R- Values", fontsize=10)
    ax.set_xlim(0,)
    ax.grid()
    ax.axhline(1, linestyle="--", color="black")
    ax.set_ylim(0,)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Reproductive number R", fontsize=10)
    if fname is not None:
        fig.tight_layout()
        plt.savefig(fname+".png", bbox_inches="tight", dpi=600)
    #plt.close()
    return
