import json
from pathlib import Path
from backend.utils.feature_extraction import bytecode, transaction, sourcecode
from backend.utils.download import download_contract_from_etherscan
from backend.utils.constants import PROJECT_ROOT, FEATURE_PATH

def extract_base_feature_from_address(address: str, save: bool = True, refresh: bool = False, output_dir="data/features"):

    address = address.lower()
    feature_path = FEATURE_PATH / f"{address}.json"

    # Use cached file if it exists
    if feature_path.exists() and not refresh:
        with open(feature_path) as f:
            return json.load(f)

    result = download_contract_from_etherscan(address, refresh=refresh)
    if not result:
        return {"error": f"Unable to fetch files for {address}"}

    txn_path, hex_path, sol_path = result

    # Feature 1: Bytecode
    if Path(hex_path).exists():
        bytecode_features = bytecode.extract_bytecode_features(hex_path)
    else:
        bytecode_features = {}

    # Feature 2: Transactions
    if Path(txn_path).exists():
        transaction_features, timeline_seq = transaction.extract_transaction_features(txn_path)
    else:
        transaction_features = {}
        timeline_seq = []

    # Feature 3: Source code
    if Path(sol_path).exists():
        sourcecode_content = sourcecode.load_sol_file(sol_path)
    else:
        sourcecode_content = ""

    # Combine all features
    combined_features = {
        **bytecode_features,
        **transaction_features,
        "timeline_sequence": timeline_seq,
        "sourcecode": sourcecode_content,
    }

    # Save (cache)
    if save:
        save_extracted_features(address, combined_features, output_dir)

    return combined_features

def save_extracted_features(address, features, output_dir="data/features"):
    address = address.lower()
    output_path = PROJECT_ROOT  / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    feature_file = output_path / f"{address.lower()}.json"

    with open(feature_file, "w") as f:
        json.dump(features, f, indent=2)

    return str(feature_file)
