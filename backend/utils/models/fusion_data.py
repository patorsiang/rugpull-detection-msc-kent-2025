import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from tensorflow.keras.models import load_model

from backend.utils.models.graph_data import GCN, form_data, val_model
from backend.utils.feature_extraction.bytecode import build_bytecode_feature_dataframe
from backend.utils.feature_extraction.transaction import build_txn_feature_dataframe
from backend.utils.feature_extraction.graph import generate_control_flow_graphs, generate_transaction_graphs, save_graph_features
from backend.utils.feature_extraction.sourcecode import build_sol_feature_dataframe
from backend.utils.models.timeline_data import extract_timeline_feature

def grouping_data(scr_path, model_path, ground_file):
    TXN_PATH = os.path.join(scr_path, 'txn')
    HEX_PATH = os.path.join(scr_path, 'hex')
    SOL_PATH = os.path.join(scr_path, 'sol')

    # Extract features
    bytecode_df, _ = build_bytecode_feature_dataframe(HEX_PATH, model_path, use_saved_model=True)
    txn_df = build_txn_feature_dataframe(TXN_PATH)
    tf_idf_df, _ = build_sol_feature_dataframe(SOL_PATH, model_path, use_saved_model=True)

    # Graph-based features
    txn_graphs = generate_transaction_graphs(TXN_PATH)
    txn_feat_df = save_graph_features(scr_path, 'txn', txn_graphs)
    txn_graphs = {addr: {'graph': graph, 'feature': txn_feat_df.loc[addr]} for addr, graph in txn_graphs.items()}
    cfg_graphs = generate_control_flow_graphs(HEX_PATH)
    cfg_feat_df = save_graph_features(scr_path, 'cfg', cfg_graphs)
    cfg_graphs = {addr: {'graph': graph, 'feature': cfg_feat_df.loc[addr]} for addr, graph in cfg_graphs.items()}

    # Timeline features (for GRU)
    ts_timeline = extract_timeline_feature(scr_path)

    # Load labels
    ground_df = pd.read_csv(os.path.join(scr_path, ground_file), index_col=0)

    # Grouped data per address
    feature_by_address = {}

    for address in ground_df.index:
        feature_by_address[address] = {
            "byte": bytecode_df.loc[address] if address in bytecode_df.index else None,
            "txn": txn_df.loc[address] if address in txn_df.index else None,
            "code": tf_idf_df.loc[address] if address in tf_idf_df.index else None,
            "txn_graph": txn_graphs.get(address, None),
            "cfg_graph": cfg_graphs.get(address, None),
            "gru": ts_timeline.get(address, None),
            "label": ground_df.loc[address].tolist()
        }

    label_cols = list(ground_df.columns)

    return feature_by_address, ground_df, label_cols

def predict_by_model_fusion(model_path, feature, label_cols, final_threshold=0.9):
    # === Load models + weights ===
    models = {
        'byte': pickle.load(open(os.path.join(model_path, 'byte.pkl'), 'rb')),
        'code': pickle.load(open(os.path.join(model_path, 'code.pkl'), 'rb')),
        'txn': pickle.load(open(os.path.join(model_path, 'txn.pkl'), 'rb'))
    }

    time_model = load_model(os.path.join(model_path, 'gru_txn_model.keras'))
    time_ext = json.load(open(os.path.join(model_path, 'gru_txn_extension.json'), 'r'))

    txn_ext = json.load(open(os.path.join(model_path, 'txn_best_params.json'), 'r'))

    label_len = len(label_cols)

    txn_thresholds = np.array(txn_ext.get('thresholds', [0.5]*label_len))
    time_thresholds = np.array(time_ext.get('thresholds', [0.5]*label_len))

    txn_gcn_model = GCN(
        in_channels=txn_ext['in_channels'],
        hidden=txn_ext['hidden_dim'],
        out_channels=txn_ext['out_channels'],
        dropout=txn_ext['dropout']
    )
    txn_gcn_model.load_state_dict(torch.load(os.path.join(model_path, "txn_model.pth")))

    cfg_ext = json.load(open(os.path.join(model_path, 'cfg_best_params.json'), 'r'))
    cfg_thresholds = np.array(cfg_ext.get('thresholds', [0.5]*label_len))
    cfg_gcn_model = GCN(
        in_channels=cfg_ext['in_channels'],
        hidden=cfg_ext['hidden_dim'],
        out_channels=cfg_ext['out_channels'],
        dropout=cfg_ext['dropout']
    )
    cfg_gcn_model.load_state_dict(torch.load(os.path.join(model_path, "cfg_model.pth")))

    # === Prepare predictions per model ===
    available_preds = {
        'byte': {},
        'code': {},
        'txn': {},
        'gru': {},
        'txn_gcn': {},
        'cfg_gcn': {}
    }

    for addr, data in feature.items():
        # Tabular models
        for name in ['byte', 'code', 'txn']:
            if data[name] is not None:
                model = models[name]['model']
                expected_cols = models[name].get('feature_cols')
                x = data[name].reindex(expected_cols, fill_value=0).values.reshape(1, -1)
                y_pred = model.predict(x)
                available_preds[name][addr] = y_pred[0]

        # GRU (timeline)
        if data['gru'] is not None:
            x = data['gru'].reshape(1, 500, -1)
            y_prob = time_model.predict(x, verbose=0)[0]
            y_pred = (y_prob > time_thresholds).astype(int)
            available_preds['gru'][addr] = y_pred[0]

        # Transaction GCN
        if data['txn_graph'] is not None:
            graph = data['cfg_graph']['graph']
            feature = data['cfg_graph']['feature']
            _, y_prob, _ = val_model(cfg_gcn_model, DataLoader([form_data(graph, feature)]))
            y_pred = (y_prob > txn_thresholds).astype(int)
            available_preds['txn_gcn'][addr] = y_pred[0]

        # CFG GCN
        if data['cfg_graph'] is not None:
            graph = data['cfg_graph']['graph']
            feature = data['cfg_graph']['feature']
            _, y_prob, _ = val_model(cfg_gcn_model, DataLoader([form_data(graph, feature)]))
            y_pred = (y_prob > cfg_thresholds).astype(int)
            available_preds['cfg_gcn'][addr] = y_pred[0]

    # === Per-model weights ===
    default_weight = [0.5] * label_len
    weights_cases = {
        'byte': np.array(models['byte'].get('weights', default_weight)),
        'code': np.array(models['code'].get('weights', default_weight)),
        'txn': np.array(models['txn'].get('weights', default_weight)),
        'gru': np.array(time_ext.get('weights', default_weight)),
        'txn_gcn': np.array(txn_ext.get('weights', default_weight)),
        'cfg_gcn': np.array(cfg_ext.get('weights', default_weight)),
    }

    final_preds = {}
    C_probs = {}

    for addr in set.union(*[set(preds.keys()) for preds in available_preds.values()]):
        weighted_sum = np.zeros(label_len)
        total_weight = np.zeros(label_len)

        for model_name, preds in available_preds.items():
            if addr in preds:
                model_weights = weights_cases[model_name]
                prediction = preds[addr]
                weighted_sum += model_weights * prediction
                total_weight += model_weights

        fused_probs = (weighted_sum / np.maximum(total_weight, 1e-8))
        fused = fused_probs > final_threshold
        C_probs[addr] = fused_probs
        final_preds[addr] = fused.astype(int)

    common_addrs = list(final_preds.keys())
    y_pred = np.array([final_preds[addr] for addr in common_addrs])
    df = pd.DataFrame(y_pred, columns=label_cols)
    df.insert(0, 'Address', common_addrs)
    df = df.set_index('Address')

    return df
