from pathlib import Path
from backend.utils.feature_extraction import bytecode, transaction, sourcecode
from backend.utils.download import download_contract_from_etherscan

def extract_base_feature_from_address(address: str):
    result = download_contract_from_etherscan(address)
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
        transaction_features = transaction.extract_transaction_features(txn_path)
    else:
        transaction_features = {}

    # Feature 3: Source code
    if Path(sol_path).exists():
        sourcecode_content = sourcecode.load_sol_file(sol_path)
    else:
        sourcecode_content = ""

    # Combine all features
    combined_features = {
        'bytecode': bytecode_features,
        'transaction': transaction_features,
        "sourcecode": sourcecode_content
    }

    return combined_features
