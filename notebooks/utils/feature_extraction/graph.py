import os
import pickle
import pandas as pd
import networkx as nx
from pathlib import Path
from evm_cfg_builder.cfg.cfg import CFG

from bytecode import load_bytecode
from transaction import load_transaction

def extract_graph_features(addr, G):
    return {
        "Address": addr,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        "density": nx.density(G),
        "connected_components": nx.number_weakly_connected_components(G),
        "avg_clustering": nx.average_clustering(G.to_undirected())
    }

def cfg_to_nx(cfg):
    G = nx.DiGraph()
    for bb in cfg.basic_blocks:
        G.add_node(bb.start.pc)  # You could also use bb.idx or bb.start.offset
        for out in bb.all_outgoing_basic_blocks:
            G.add_edge(bb.start.pc, out.start.pc)
    return G

def get_cfg_from_file(hex_file):
    bytecode = load_bytecode(hex_file)
    if bytecode == '0x':
        return
    return CFG(bytecode)

def generate_control_flow_graphs(hex_dir, address=None):
    all_files = list(Path(hex_dir).glob(f'{address if address is not None else '*'}.hex'))

    graphs = dict()

    for file in all_files:
        address = file.stem.lower()  # remove '.hex' and lowercase
        print(f"Processing control flow of {address}")
        try:
            cfg = get_cfg_from_file(file)
            if cfg is None:
                continue
            nx_graph = cfg_to_nx(cfg)
            graphs[address] = nx_graph
        except Exception as e:
            print(f"⚠️ Error in {file.name}: {e}")
            continue
    return graphs

def save_graphs(out_dir, name, graphs):
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{name}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(graphs, f)
    return filename

def save_graph_features(out_dir, name, graphs):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for addr, G in graphs.items():
        features = extract_graph_features(addr, G)
        rows.append(features)

    if not rows:
        print(f"[WARN] No features extracted for {name}, skipping save.")
        return pd.DataFrame()  # or return None
    df = pd.DataFrame(rows).set_index('Address')
    df.fillna(0)
    df.to_csv(os.path.join(out_dir, f"{name}_graph_features.csv"))
    save_graphs(out_dir, name, graphs)
    return df

def txn_to_nx(txn):
    G = nx.DiGraph()
    for tx in txn:
        from_addr = tx.get("from", "").lower()
        to_addr = tx.get("to", "").lower()
        tx_hash = tx.get("hash", "")
        gas = int(tx.get("gasUsed", 0))

        if not from_addr:
            continue

        if to_addr == "":
            # Contract creation
            contract_addr = tx.get("contractAddress", f"created_{tx_hash}")
            G.add_node(contract_addr)
            G.add_edge(from_addr, contract_addr, tx_hash=tx_hash, label="contract_creation", gas=gas)
        elif to_addr:
            # Normal transaction
            G.add_edge(from_addr, to_addr, tx_hash=tx_hash, label="transaction", gas=gas)
    return G


def generate_transaction_graphs(json_dir, address=None):
    all_files = list(Path(json_dir).glob(f'{address if address is not None else '*'}.json'))

    graphs = dict()

    for file in all_files:
        try:
            address, _, transactions, _  = load_transaction(file)
            if len(transactions) > 0:
                nx_graph = txn_to_nx(transactions)
                graphs[address] = nx_graph
        except Exception as e:
            print(f"error on {file.name}: {e}")
    return graphs

