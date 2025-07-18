import networkx as nx
from evm_cfg_builder.cfg.cfg import CFG
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
import os

import sys
sys.path.append(str(Path.cwd().parents[1]))
from scripts.utils import load_bytecode

def cfg_to_nx(cfg):
    G = nx.DiGraph()
    for bb in cfg.basic_blocks:
        G.add_node(bb.start.pc)  # You could also use bb.idx or bb.start.offset
        for out in bb.all_outgoing_basic_blocks:
            G.add_edge(bb.start.pc, out.start.pc)
    return G

def get_cfg_from_file(hex_file):
    bytecode = load_bytecode(hex_file)
    return CFG(bytecode)

def extract_graph_features(G):
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "density": nx.density(G),
        "connected_components": nx.number_weakly_connected_components(G),
        "avg_clustering": nx.average_clustering(G.to_undirected())
    }

def get_graphs_stat_from_files(files):
    records = []
    for file in tqdm(files):
        address = file.stem.lower()  # remove '.hex' and lowercase
        cfg = get_cfg_from_file(file)
        nx_graph = cfg_to_nx(cfg)
        feats = extract_graph_features(nx_graph)
        feats['address'] = address
        records.append(feats)

    return pd.DataFrame(records).fillna(0).set_index('address')

def save_graphs_and_labels_from_files(files, labels, dest_path, file_name):
    graphs = []

    for file in tqdm(files):
        cfg = get_cfg_from_file(file)
        nx_graph = cfg_to_nx(cfg)
        graphs.append(nx_graph)

    with open(os.path.join(dest_path, file_name), "wb") as f:
        pickle.dump((graphs, labels), f)

    print(f'saved {file_name}')
