import time
from pathlib import Path
from backend.utils.data_loader import (
    get_info_by_contract_addr,
    save_bytecode_by_contract_addr,
    get_bytecode_by_contract_addr,
    save_transactions_by_contract_addr,
    get_source_code_by_contract_addr,
    save_sol_by_contract_addr
)
from backend.utils.logger import get_logger

def download_contract_from_etherscan(address: str, tmp_path: str = 'interim', refresh: bool = False):
    address = address.lower()
    logger = get_logger('download_contract_from_etherscan')
    logger.info(f"Searching {address} ...")

    chains = {
        "ETH": 1, "BSC": 56, "Polygon": 137, "Arbitrum": 42161,
        "Fantom": 146, "BASE": 8453, "AVAX": 43114,
        "OP.ETH": 10, "Cronos": 25, "Blast": 81457
    }

    PATH = Path(__file__).resolve().parents[2]
    DATA_PATH = PATH / "data"
    LABELED_PATH = DATA_PATH / "labeled"
    UNLABELED_PATH = DATA_PATH / "unlabeled"
    TMP_PATH = DATA_PATH / tmp_path

    def build_paths(base):
        return (
            base / 'txn' / f"{address}.json",
            base / 'hex' / f"{address}.hex",
            base / 'sol' / f"{address}.sol"
        )

    if not refresh:
        # 1. Check Labeled
        txn_path, hex_path, sol_path = build_paths(LABELED_PATH)
        if txn_path.exists() or hex_path.exists() or sol_path.exists():
            logger.info(f"Found {address} in labeled dataset.")
            return [txn_path, hex_path, sol_path]

        # 2. Check Unlabeled
        txn_path, hex_path, sol_path = build_paths(UNLABELED_PATH)
        if txn_path.exists() or hex_path.exists() or sol_path.exists():
            logger.info(f"Found {address} in unlabeled dataset.")
            return [txn_path, hex_path, sol_path]

    # 3. Prepare interim folders
    TXN_PATH, HEX_PATH, SOL_PATH = TMP_PATH / 'txn', TMP_PATH / 'hex', TMP_PATH / 'sol'
    for path in [TXN_PATH, HEX_PATH, SOL_PATH]:
        path.mkdir(parents=True, exist_ok=True)

    txn_path, hex_path, sol_path = build_paths(TMP_PATH)
    has_txn, has_hex, has_sol = txn_path.exists(), hex_path.exists(), sol_path.exists()

    if refresh:
        has_txn, has_hex, has_sol = False, False, False

    if has_txn and has_hex and has_sol:
        logger.info(f"Already downloaded previously.")
        return [txn_path, hex_path, sol_path]

    # 4. Attempt download from multiple chains
    for chain_name, chain_id in chains.items():
        logger.info(f"Checking {chain_name} ({chain_id}) ...")

        try:
            if not has_txn:
                info = get_info_by_contract_addr(address, chain_id)
                if isinstance(info, dict):
                    save_transactions_by_contract_addr(TXN_PATH, address, info)
                    has_txn = True

                    if 'creator' in info and 'creationBytecode' in info['creator']:
                        save_bytecode_by_contract_addr(HEX_PATH, address, info['creator']['creationBytecode'])
                        has_hex = True
                time.sleep(0.3)

            if not has_hex:
                bytecode = get_bytecode_by_contract_addr(address, chain_id)
                if bytecode:
                    save_bytecode_by_contract_addr(HEX_PATH, address, bytecode)
                    has_hex = True
                time.sleep(0.3)

            if not has_sol:
                source = get_source_code_by_contract_addr(address, chain_id)
                if source and 'SourceCode' in source:
                    save_sol_by_contract_addr(SOL_PATH, address, source['SourceCode'])
                    has_sol = True
                time.sleep(0.3)

        except Exception as e:
            logger.warning(f"Failed on {chain_name}: {e}")
            continue

        # Exit early if any are found
        if has_txn or has_hex or has_sol:
            break

    return [txn_path, hex_path, sol_path] if has_txn or has_hex or has_sol else None
