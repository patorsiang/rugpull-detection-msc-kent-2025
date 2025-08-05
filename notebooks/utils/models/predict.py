import os
import time
import pandas as pd

from backend.utils.data_loader import (
  get_info_by_contract_addr,
  save_bytecode_by_contract_addr,
  get_bytecode_by_contract_addr,
  save_transactions_by_contract_addr,
  get_source_code_by_contract_addr,
  save_sol_by_contract_addr
)

from backend.utils.models.fusion_data import grouping_data, predict_by_model_fusion

def predict_by_addr(address, tmp_path='interim', model_path='models', threshold=0.6):
    address = address.lower()
    print(f"Searching {address} ...")
    chains = {
        "ETH": 1,
        "BSC": 56,
        "Polygon": 137,
        "Arbitrum": 42161,
        "Fantom": 146,
        "Sonic": 146,
        "Avax": 43114,
        "BASE": 8453,
        "AVAX": 43114,
        "Cchain": 43114,
        "OP.ETH": 10,
        "SnowTrace": 43114,
        "Heco": 128,
        "Cronos": 25,
        "Blast": 81457
    }

    TXN_PATH = os.path.join(tmp_path, 'txn')
    HEX_PATH = os.path.join(tmp_path, 'hex')
    SOL_PATH = os.path.join(tmp_path, 'sol')

    for chain_name, chain_id in chains.items():
        print(f"Checking {chain_name} ...")
        has_txn = os.path.exists(os.path.join(TXN_PATH, f'{address}.json'))
        has_hex = os.path.exists(os.path.join(HEX_PATH, f'{address}.hex'))
        has_sol = os.path.exists(os.path.join(SOL_PATH, f'{address}.sol'))
        if not has_txn:
            info = get_info_by_contract_addr(address, chain_id)
            if info is not dict():
                save_transactions_by_contract_addr(TXN_PATH, address, info)
                has_txn = True
            if 'creationBytecode' in info.get('creator'):
                save_bytecode_by_contract_addr(HEX_PATH, address, info['creator']['creationBytecode'])
                has_hex = True
            time.sleep(0.5)
        if not has_hex:
            bytecode = get_bytecode_by_contract_addr(address, chain_id)
            if bytecode != "":
                save_bytecode_by_contract_addr(HEX_PATH, address, bytecode)
                has_hex = True
            time.sleep(0.5)
        if not has_sol:
            source = get_source_code_by_contract_addr(address, chain_id)
            if 'SourceCode' in source:
                save_sol_by_contract_addr(SOL_PATH, address, source['SourceCode'])
                has_sol = True
            time.sleep(0.5)
        if has_txn or has_hex or has_sol:
            break

    filename = os.path.join(tmp_path, f"{address}.csv")
    # Create DataFrame
    pd.DataFrame([{
        'Address': address,
        'Mint': -1,
        'Leak': -1,
        'Limit': -1
    }]).to_csv(filename, index=False)

    feature, _, label_cols = grouping_data(tmp_path, model_path, filename, address=address)
    print(feature)
    preds_df = predict_by_model_fusion(model_path, feature, label_cols, threshold)
    preds_df.to_csv(filename)
    print(f"Predictions saved to {filename}")
    return preds_df

