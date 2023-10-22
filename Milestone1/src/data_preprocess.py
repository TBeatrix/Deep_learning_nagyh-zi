import re
#!pip install nodevectors== 0.1.0
import networkx as nx
import gensim

from nodevectors import Node2Vec
import matplotlib.pyplot as plt

import csrgraph as cg
from nodevectors.embedders import BaseNodeEmbedder
from sklearn.model_selection import train_test_split
import subprocess
import numba
import numpy as np
import pandas as pd
import time
import warnings


# Gensim triggers automatic useless warnings for windows users...
warnings.simplefilter("ignore", category=UserWarning)

warnings.simplefilter("default", category=UserWarning)


class Node2Vec(BaseNodeEmbedder):
    def __init__(
        self,
        n_components=32,
        walklen=30,
        epochs=20,
        return_weight=1.,
        neighbor_weight=1.,
        threads=0,
        keep_walks=False,
        verbose=True,
        w2vparams={"window":10, "negative":5, "iter":10,
                   "batch_words":128}):
      
       
      
        if type(threads) is not int:
            raise ValueError("Threads argument must be an int!")
        if walklen < 1 or epochs < 1:
            raise ValueError("Walklen and epochs arguments must be > 1")
        self.n_components = n_components
        self.walklen = walklen
        self.epochs = epochs
        self.keep_walks = keep_walks
        if 'size' in w2vparams.keys():
            raise AttributeError("Embedding dimensions should not be set "
                + "through w2v parameters, but through n_components")
        self.w2vparams = w2vparams
        self.return_weight = return_weight
        self.neighbor_weight = neighbor_weight
        if threads == 0:
            threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
        self.threads = threads
        w2vparams['workers'] = threads
        self.verbose = verbose

    def fit(self, G):
     
       
       
        if not isinstance(G, cg.csrgraph):
            G = cg.csrgraph(G, threads=self.threads)
        if G.threads != self.threads:
            G.set_threads(self.threads)
        # Because networkx graphs are actually iterables of their nodes
        #   we do list(G) to avoid networkx 1.X vs 2.X errors
        node_names = G.names
        if type(node_names[0]) not in [int, str, np.int32, np.uint32,
                                       np.int64, np.uint64]:
            raise ValueError("Graph node names must be int or str!")
        # Adjacency matrix
        walks_t = time.time()
        if self.verbose:
            print("Making walks...", end=" ")
        self.walks = G.random_walks(walklen=self.walklen,
                                    epochs=self.epochs,
                                    return_weight=self.return_weight,
                                    neighbor_weight=self.neighbor_weight)
        if self.verbose:
            print(f"Done, T={time.time() - walks_t:.2f}")
            print("Mapping Walk Names...", end=" ")
        map_t = time.time()
        self.walks = pd.DataFrame(self.walks)
        # Map nodeId -> node name
        node_dict = dict(zip(np.arange(len(node_names)), node_names))
        for col in self.walks.columns:
            self.walks[col] = self.walks[col].map(node_dict).astype(str)
        # Somehow gensim only trains on this list iterator
        # it silently mistrains on array input
        self.walks = [list(x) for x in self.walks.itertuples(False, None)]
        if self.verbose:
            print(f"Done, T={time.time() - map_t:.2f}")
            print("Training W2V...", end=" ")
            if gensim.models.word2vec.FAST_VERSION < 1:
                print("WARNING: gensim word2vec version is unoptimized"
                    "Try version 3.6 if on windows, versions 3.7 "
                    "and 3.8 have had issues")
        w2v_t = time.time()
        # Train gensim word2vec model on random walks
        self.model = gensim.models.Word2Vec(
            sentences=self.walks,
            vector_size=self.n_components, #!!!!
            **self.w2vparams)
        if not self.keep_walks:
            del self.walks
        if self.verbose:
            print(f"Done, T={time.time() - w2v_t:.2f}")

    def fit_transform(self, G):
      
        if not isinstance(G, cg.csrgraph):
            G = cg.csrgraph(G, threads=self.threads)
        self.fit(G)
        w = np.array(
            pd.DataFrame.from_records(
            pd.Series(np.arange(len(G.nodes())))
              .apply(self.predict)
              .values)
        )
        return w

    def predict(self, node_name):
      
        # current hack to work around word2vec problem
        # ints need to be str -_-
        if type(node_name) is not str:
            node_name = str(node_name)
        return self.model.wv.__getitem__(node_name)

    def save_vectors(self, out_file):

     
        self.model.wv.save_word2vec_format(out_file)

    def load_vectors(self, out_file):
       
        self.model = gensim.wv.load_word2vec_format(out_file)
     


def get_ego_indexes(files):
  pattern = re.compile(r'\d+')
  ego_indexes = [int(pattern.search(s).group()) for s in files if pattern.search(s)]
  ego_indexes = sorted(set(ego_indexes))
  return ego_indexes

class GraphData:
  edge_index = []
  edge_list = []
  nodes = []
  X = []
  circles = []
  X_names = []
  embeddings = []
  train_mask = []
  test_mask = []
  val_mask = []

  def __init__(self, edges, edge_list, X, circles, X_names, embeddings, nodes, train_mask, test_mask, val_mask):
    self.edge_index = edges
    self.edge_list = edge_list
    self.X = X
    self.circles = circles
    self.X_names = X_names
    self.embeddings = embeddings
    self.nodes = nodes
    self.train_mask = train_mask
    self.test_mask = test_mask
    self.val_mask = val_mask
    

def visualize(g, title="Graph", edge='blue'):
    pos = nx.kamada_kawai_layout(g)
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.axis('on')
    nx.draw_networkx(g, pos=pos, node_size=10,
                     arrows=False, width=1, style='solid', with_labels= False)
    plt.savefig('src/graph.png')
    

if __name__ == '__main__':
  nx.adj_matrix = nx.adjacency_matrix
  with open('facebook/indexes.txt', 'r') as f:
    Fb_ego_indexes= [int(line.strip()) for line in f]

  Facebook_graphs = []

  for index in range(len(Fb_ego_indexes)-1):
    f = open(f'facebook/{Fb_ego_indexes[index]}.edges', "rt")
    edges_str = f.readlines()
    edges = [list(map(int, item.strip().split())) for item in edges_str]
    # Transpose the data to get desired format
    edges = [list(row) for row in zip(*edges)]

    edge_list =   [(int(x.split()[0]), int(x.split()[1])) for x in edges_str]
    f = open(f'facebook/{Fb_ego_indexes[index]}.circles', "rt")
    circles_str = f.readlines()
    circles = [list(map(int, item.split('\t')[1:])) for item in circles_str]

    f = open(f'facebook/{Fb_ego_indexes[index]}.featnames', "rt")
    feat_name_str = f.readlines()
    X_names = [item.split(';anonymized feature')[0].split(';') for item in feat_name_str]

    f = open(f'facebook/{Fb_ego_indexes[index]}.egofeat', "rt")
    feat_str = f.readlines()
    X = []
    X.append(list(map(int, feat_str[0].split()[1:])))

    f = open(f'facebook/{Fb_ego_indexes[index]}.feat', "rt")
    feat_str = f.readlines()
    X.append([list(map(int, item.split()[1:])) for item in feat_str])

    # Node2Vec expects an NX graph
    G  = nx.from_edgelist(edge_list)
    # Node2Vec model specification
    n2v = Node2Vec(n_components=64, walklen=10, epochs=50, return_weight=1.0, neighbor_weight=1.0, threads=0, w2vparams={'window': 4, 'negative': 5, 'epochs':10, 'ns_exponent': 0.5, 'batch_words': 128})
    # Fit and get the embedding
    print(f"Create Node2Vec embeddings for graph {index+1}")
    n2v.fit(G)
    nodes = G.nodes()
    embeddings = []
    for node in nodes:
      embeddings.append(n2v.predict(node))

    idx = np.arange(len(nodes))
    train_idx, test_idx = train_test_split(idx, train_size=0.8,  random_state=17)
    val_idx, test_idx = train_test_split(test_idx, train_size=0.5,  random_state=17)

    Facebook_graphs.append(GraphData(edges, edge_list, X, circles, X_names, embeddings, nodes, train_idx, test_idx, val_idx))
  
  Edge_graph = nx.from_edgelist(Facebook_graphs[0].edge_list)
  visualize( Edge_graph, "First graph")
  print("train indexes of the first graph: \n", Facebook_graphs[0].train_mask)
  print("test indexes of the first graph: \n", Facebook_graphs[0].test_mask)
  print("validation indexes of the first graph: \n", Facebook_graphs[0].val_mask)
  
