import time

from backend.utils.data_loader import (
    get_info_by_contract_addr,
    save_bytecode_by_contract_addr,
    get_bytecode_by_contract_addr,
    save_transactions_by_contract_addr,
    get_source_code_by_contract_addr,
    save_sol_by_contract_addr,
)
from backend.utils.logger import logging
from backend.utils.constants import HEX_PATH, TXN_PATH, SOL_PATH
from backend.utils.etherscan_quota import quota_guard, QuotaExceeded

logger = logging.getLogger(__name__)


def download_contract_from_etherscan(address: str, refresh: bool = False):
    """
    Attempts to fetch txn, hex, and sol for the address across multiple chains.
    - If all files already exist and refresh=False -> returns existing paths.
    - If quota is exhausted and any artifact is missing -> aborts early with QuotaExceeded (returns None here).
    """
    address = address.lower()
    logger.info(f"Searching {address} ...")

    chains = {
        "ETH": 1,
        "BSC": 56,
        "Polygon": 137,
        "Arbitrum": 42161,
        "Fantom": 146,
        "BASE": 8453,
        "AVAX": 43114,
        "OP.ETH": 10,
        "Cronos": 25,
        "Blast": 81457,
    }

    txn_path = TXN_PATH / f"{address}.json"
    hex_path = HEX_PATH / f"{address}.hex"
    sol_path = SOL_PATH / f"{address}.sol"

    has_txn, has_hex, has_sol = txn_path.exists(), hex_path.exists(), sol_path.exists()

    if refresh:
        has_txn = False  # force re-fetch of transactions (often the heaviest)

    if has_txn and has_hex and has_sol:
        logger.info("Already downloaded previously.")
        return [txn_path, hex_path, sol_path]

    # If anything is missing, verify there's at least some quota left before starting.
    try:
        quota_guard.require_available_or_raise(1)
    except QuotaExceeded as e:
        logger.warning(f"Quota exhausted before download: {e}")
        return None

    for chain_name, chain_id in chains.items():
        logger.info(f"Checking {chain_name} ({chain_id}) ...")

        if not has_txn:
            try:
                info = get_info_by_contract_addr(address, chain_id)
                if isinstance(info, dict):
                    save_transactions_by_contract_addr(TXN_PATH, address, info)
                    has_txn = True

                    # Prefer creationBytecode if present
                    if "creator" in info and "creationBytecode" in info["creator"] and not has_hex:
                        save_bytecode_by_contract_addr(HEX_PATH, address, info["creator"]["creationBytecode"])
                        has_hex = True
                    time.sleep(0.2)
            except QuotaExceeded as e:
                logger.warning(f"Quota exhausted on {chain_name}: {e}")
                return None
            except Exception as e:
                logger.warning(f"Failed on {chain_name} (txn step): {e}")

        if not has_hex:
            try:
                bytecode = get_bytecode_by_contract_addr(address, chain_id)
                if bytecode:
                    save_bytecode_by_contract_addr(HEX_PATH, address, bytecode)
                    has_hex = True
                time.sleep(0.2)
            except QuotaExceeded as e:
                logger.warning(f"Quota exhausted on {chain_name}: {e}")
                return None
            except Exception as e:
                logger.warning(f"Failed on {chain_name} (hex step): {e}")

        if not has_sol:
            try:
                source = get_source_code_by_contract_addr(address, chain_id)
                if source and "SourceCode" in source:
                    save_sol_by_contract_addr(SOL_PATH, address, source["SourceCode"])
                    has_sol = True
                time.sleep(0.2)
            except QuotaExceeded as e:
                logger.warning(f"Quota exhausted on {chain_name}: {e}")
                return None
            except Exception as e:
                logger.warning(f"Failed on {chain_name} (sol step): {e}")

        if has_txn and has_hex and has_sol:
            break

    return [txn_path, hex_path, sol_path] if (has_txn or has_hex or has_sol) else None
