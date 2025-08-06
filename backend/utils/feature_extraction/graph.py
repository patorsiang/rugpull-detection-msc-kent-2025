import networkx as nx

def extract_control_flow_graph_features(cfg = None):
    if cfg is None:
        return extract_graph_features(nx.DiGraph())

    G = nx.DiGraph()
    for bb in cfg.basic_blocks:
        G.add_node(bb.start.pc)  # You could also use bb.idx or bb.start.offset
        for out in bb.all_outgoing_basic_blocks:
            G.add_edge(bb.start.pc, out.start.pc)
    return extract_graph_features(G)

def extract_transaction_graph_features(transactions: list):
    G = nx.DiGraph()
    for txn in transactions:
        from_addr = txn.get("from", "").lower()
        to_addr = txn.get("to", "").lower()
        tx_hash = txn.get("hash", "")
        gas = int(txn.get("gasUsed", 0))

        if not from_addr:
            continue

        if to_addr == "":
            # Contract creation
            contract_addr = txn.get("contractAddress", f"created_{tx_hash}")
            G.add_node(contract_addr)
            G.add_edge(from_addr, contract_addr, tx_hash=tx_hash, label="contract_creation", gas=gas)
        elif to_addr:
            # Normal transaction
            G.add_edge(from_addr, to_addr, tx_hash=tx_hash, label="transaction", gas=gas)
    return extract_graph_features(G)

def extract_graph_features(G: nx.Graph):
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        "density": nx.density(G),
        "connected_components": nx.number_weakly_connected_components(G),
        "avg_clustering": nx.average_clustering(G.to_undirected())
    }
