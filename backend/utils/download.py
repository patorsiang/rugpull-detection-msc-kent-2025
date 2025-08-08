import time
from backend.utils.data_loader import (
    get_info_by_contract_addr,
    save_bytecode_by_contract_addr,
    get_bytecode_by_contract_addr,
    save_transactions_by_contract_addr,
    get_source_code_by_contract_addr,
    save_sol_by_contract_addr
)
from backend.utils.logger import get_logger
from backend.utils.constants import HEX_PATH, TXN_PATH, SOL_PATH

def download_contract_from_etherscan(address: str, refresh: bool = False):
    address = address.lower()
    logger = get_logger('download_contract_from_etherscan')
    logger.info(f"Searching {address} ...")

    chains = {
        "ETH": 1, "BSC": 56, "Polygon": 137, "Arbitrum": 42161,
        "Fantom": 146, "BASE": 8453, "AVAX": 43114,
        "OP.ETH": 10, "Cronos": 25, "Blast": 81457
    }

    txn_path, hex_path, sol_path = TXN_PATH / f"{address}.json", HEX_PATH / f"{address}.hex", SOL_PATH / f"{address}.sol"

    has_txn, has_hex, has_sol = txn_path.exists(), hex_path.exists(), sol_path.exists()

    if refresh:
        has_txn = False

    if has_txn and has_hex and has_sol:
        logger.info(f"Already downloaded previously.")
        return [txn_path, hex_path, sol_path]

    # 4. Attempt download from multiple chains
    for chain_name, chain_id in chains.items():
        logger.info(f"Checking {chain_name} ({chain_id}) ...")


        if not has_txn:
            try:
                info = get_info_by_contract_addr(address, chain_id)
                if isinstance(info, dict):
                    save_transactions_by_contract_addr(TXN_PATH, address, info)
                    has_txn = True

                    if 'creator' in info and 'creationBytecode' in info['creator'] and not has_hex:
                        save_bytecode_by_contract_addr(HEX_PATH, address, info['creator']['creationBytecode'])
                        has_hex = True
                    time.sleep(0.3)
            except Exception as e:
                logger.warning(f"Failed on {chain_name}: {e}")

        if not has_hex:
            try:
                bytecode = get_bytecode_by_contract_addr(address, chain_id)
                if bytecode:
                    save_bytecode_by_contract_addr(HEX_PATH, address, bytecode)
                    has_hex = True
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"Failed on {chain_name}: {e}")

        if not has_sol:
            try:
                source = get_source_code_by_contract_addr(address, chain_id)
                if source and 'SourceCode' in source:
                    save_sol_by_contract_addr(SOL_PATH, address, source['SourceCode'])
                    has_sol = True
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"Failed on {chain_name}: {e}")

        if has_txn and has_hex and has_sol:
            break


    return [txn_path, hex_path, sol_path] if has_txn or has_hex or has_sol else None
