import json
from pathlib import Path
from typing import Dict, Tuple, Any
from backend.utils.feature_extraction import bytecode, transaction, sourcecode
from backend.utils.download import download_contract_from_etherscan
from backend.utils.constants import PROJECT_ROOT, FEATURE_PATH
from backend.utils.logger import logging

logger = logging.getLogger(__name__)

def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.exception(f"error in {fn.__name__}: {e}")
        return None

def extract_base_feature_from_address(
    address: str,
    save: bool = True,
    refresh: bool = False,
    output_dir: str = "data/features",
) -> Dict[str, Any]:
    """
    Extracts/loads:
      - bytecode features
      - transaction features (+ timeline sequence)
      - raw source code string
    Caches to FEATURE_PATH/{address}.json
    """
    address = address.lower()
    feature_path = FEATURE_PATH / f"{address}.json"

    # Use cached file if it exists
    if feature_path.exists() and not refresh:
        try:
            with open(feature_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"failed to read cache for {address}: {e}")

    # Download artifacts
    result: Tuple[str, str, str] | None = _safe_call(download_contract_from_etherscan, address, refresh=refresh)
    if not result or not isinstance(result, (list, tuple)) or len(result) != 3:
        logger.error(f"unable to fetch files for {address}")
        combined_features = {"timeline_sequence": [], "sourcecode": "", "opcode_sequence": ""}
        if save:
            save_extracted_features(address, combined_features, output_dir)
        return combined_features

    txn_path, hex_path, sol_path = result

    # Bytecode features
    bytecode_features = {}
    if Path(hex_path).exists():
        bc = _safe_call(bytecode.extract_bytecode_features, hex_path)
        if isinstance(bc, dict):
            bytecode_features = bc
        else:
            logger.warning(f"bytecode features invalid for {address}")

    # Transaction features (+ timeline)
    transaction_features, timeline_seq = {}, []
    if Path(txn_path).exists():
        tx_res = _safe_call(transaction.extract_transaction_features, txn_path)
        if isinstance(tx_res, tuple) and len(tx_res) == 2:
            transaction_features, timeline_seq = tx_res
        else:
            logger.warning(f"transaction features invalid for {address}")

    # Source code
    sourcecode_content = ""
    if Path(sol_path).exists():
        sc = _safe_call(sourcecode.load_sol_file, sol_path)
        if isinstance(sc, str):
            sourcecode_content = sc

    combined_features = {
        **bytecode_features,
        **transaction_features,
        "timeline_sequence": timeline_seq or [],
        "sourcecode": sourcecode_content or "",
    }

    if save:
        save_extracted_features(address, combined_features, output_dir)

    return combined_features

def save_extracted_features(address: str, features: Dict[str, Any], output_dir: str = "data/features"):
    address = address.lower()
    output_path = PROJECT_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    feature_file = output_path / f"{address}.json"
    try:
        with open(feature_file, "w") as f:
            json.dump(features, f, indent=2)
    except Exception as e:
        logger.exception(f"failed to save features for {address}: {e}")
    return str(feature_file)
