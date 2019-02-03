import pickle
import matplotlib
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import os

def denoise_graph(adj, weights, node_idx, feat=None, label=None, threshold=0.1):
    num_nodes = adj.shape[-1]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    print('num nodes: ', G.number_of_nodes())
    G.node[node_idx]['self'] = 1
    if feat is not None:
        for node in G.nodes():
            G.node[node]['feat'] = feat[node]
    if label is not None:
        for node in G.nodes():
            G.node[node]['label'] = label[node] 
    weighted_edge_list = [(i, j, adj[i, j]) for i in range(num_nodes) for j in range(num_nodes) if
            adj[i,j] > threshold and weights[i,j] > threshold]
    G.add_weighted_edges_from(weighted_edge_list)
    # return G
    Gc = max(nx.connected_component_subgraphs(G), key=len) 
    return Gc

def extract_neighborhood(adj, features, labels, weights, node_idx, name, threshold, n_hops=3):
    def _neighborhoods():
        hop_adj = power_adj = adj
        for i in range(n_hops-1):
            power_adj = np.matmul(power_adj, adj)
            hop_adj = hop_adj + power_adj
            hop_adj = (hop_adj > 0).astype(int)
        return hop_adj
    
    neighborhoods = _neighborhoods()
    neighbors_adj_row = neighborhoods[node_idx, :].A
    neighbors_adj_row = neighbors_adj_row[0]

    # index of the query node in the new adj
    node_idx_new = sum(neighbors_adj_row[:node_idx])
    neighbors = np.nonzero(neighbors_adj_row)[0]
    sub_adj = adj[neighbors][:, neighbors]

    sub_weight = weights[neighbors][:, neighbors]
    sub_feat = np.array(features)[neighbors]
    sub_label = np.array(labels)[neighbors]
    # print('==>', labels, neighbors, sub_label)

    name = os.path.splitext(name)[0] + '_neighbor_' + str(node_idx)
    adj, features, labels, weights, node_idx = sub_adj, sub_feat, sub_label, sub_weight, node_idx_new

    D = denoise_graph(adj, weights, node_idx, feat=features, label=labels, threshold=threshold)
    return D, adj, features, labels, weights, node_idx, name

def visualize(
    data_pkl,
    weights_pkl,
    node_idx=None,
    identify_self=True,
    nodecolor='label',
    epoch=0,
    fig_size=(4,3),
    dpi=300,
    name='full',
    threshold=0.5):
    
    data = pickle.load(open(data_pkl, 'rb'))
    adj, features, labels = data['adj'], data['features'], data['labels']

    weights = pickle.load(open(weights_pkl, 'rb'))
    weights = weights.detach().numpy()
    weights = (weights + weights.T) / 2
    
    D = nx.from_numpy_matrix(adj)
    for n in D.nodes():
        D.node[n]['feat'] = features[n]
        D.node[n]['label'] = labels[n]

    # visualize neighborhood
    if node_idx != None:
        print('==> visualize neighborhood of node {}'.format(node_idx))
        D, adj, features, labels, weights, node_idx, name = extract_neighborhood(adj, features, labels, weights, node_idx, name, threshold)
    else:
        print('==> visualize all')

    cmap = plt.get_cmap('Set1')
    plt.switch_backend('agg')
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    edge_colors = [weights[r][c] for r,c in D.edges if weights[r][c]]
    node_colors = [i+1 for i in labels]
    node_colors = []
    for n in D.nodes():
        node_colors.append(D.node[n]['label'] + 1)

    plt.switch_backend('agg')
    fig = plt.figure(figsize=fig_size, dpi=dpi)
#     nx.draw(D, pos=nx.spring_layout(D), with_labels=False, font_size=4,
#             node_color=node_colors, vmin=0, vmax=8, cmap=cmap,
#             edge_color=edge_colors, edge_cmap=plt.get_cmap('Greys'),
#             edge_vmin=0, edge_vmax=np.mean(edge_colors),
#             width=0.5, node_size=3,
#             alpha=0.7)
    vmax = 8
    pos_layout = nx.spring_layout(D)
    nx.draw(D, pos=nx.kamada_kawai_layout(D), with_labels=False, font_size=4, labels=None,
            node_color=node_colors, vmin=0, vmax=vmax, cmap=cmap,
            edge_color=edge_colors, edge_cmap=plt.get_cmap('Greys'), 
            edge_vmin=0.0,
            edge_vmax=np.mean(edge_colors),
            width=1.0, node_size=50,
            alpha=0.8)
    
    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()
    os.makedirs('log', exist_ok=True)
    save_path = os.path.join('log', name+'.png')
    plt.savefig(save_path)
    print('==> Save fig to {}'.format(save_path))


if __name__ == '__main__':
    # # global
    # visualize(data_pkl='./data/gnn/data_syn2.pkl',
    #         weights_pkl='./weights/weights_syn2.pkl')

    # local
    for node_idx in range(400, 411, 5):
        visualize(data_pkl='./data/gnn/data_syn2.pkl',
                  weights_pkl='./weights/weights_syn2.pkl',
                  node_idx=node_idx,
                  threshold=0.125)


