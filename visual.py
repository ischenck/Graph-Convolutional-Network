import numpy as np
import pandas as pd
import holoviews as hv
import networkx as nx
from holoviews import opts
from holoviews.operation.datashader import datashade, bundle_graph


from helpers import load_data 

hv.extension('bokeh')


defaults = dict(width=1000, height=1000, padding=0.1)
hv.opts.defaults(
    opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))

def draw_graph(G, label, cmap): 
    print(f'Shading {label}')   
    shaded = (datashade(G, normalization='linear', width=1000, height=1000) * G.nodes).opts(
        opts.Nodes(
                    color=label, 
                    width=1000, 
                    cmap=cmap, 
                    legend_position='right'
                  )
        )

    hv.save(shaded, f'graphs/png/{label}.png')
    hv.save(shaded, f'graphs/html/{label}.html')

if __name__ == '__main__':
    print('Loading')
    (adj, features, labels, 
     index_train, index_val, 
     index_test) = load_data('./data', 'cora')

    to_dict = lambda l: {i: {'class': int(v)} for i, v in enumerate(l)}

    print('Converting to NetworkX')
    G = nx.from_numpy_matrix(adj.to_dense().numpy())

    print('Creating Labels')
    labels_no_test = labels.clone()
    labels_no_test[index_test] = -1

    labels_only_train = labels.clone()
    labels_only_train[index_test] = -1
    labels_only_train[index_val] = -1

    labels_no_labels = labels.clone()
    labels_no_labels[:] = -1

    labels = {i: {'labels_all': labels[i],
                  'labels_no_test': labels_no_test[i],
                  'labels_only_train': labels_only_train[i],
                  'labels_no_labels': labels_no_labels[i]} 
              for i in range(len(labels))}

    print('Setting attributes')
    nx.set_node_attributes(G, labels)

    print('Bundling')
    G = bundle_graph(hv.Graph.from_networkx(G, nx.layout.spring_layout).opts(tools=['hover']))

    cmap = ['#0000ff', '#8888ff', '#ffffff', '#ff8888', '#ff0000', '#00ffff', '#ffff00']
    draw_graph(G, 'labels_all', cmap)
    draw_graph(G, 'labels_no_test', ['#222222'] + cmap)
    draw_graph(G, 'labels_only_train', ['#222222'] + cmap)
    draw_graph(G, 'labels_no_labels', ['#222222'])

