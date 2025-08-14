import os
import json
import time
import requests

from backend.utils.logger import logging
from backend.utils.etherscan_quota import quota_guard, QuotaExceeded
from backend.utils.constants import ETHERSCAN_API_KEY, ETHERSCAN_BASE_URL, ETHERSCAN_HTTP_TIMEOUT

logger = logging.getLogger(__name__)


def _get(params: dict) -> dict:
    """
    One-shot GET with basic error handling + quota guard.
    Raises QuotaExceeded if daily quota is exhausted.
    """
    quota_guard.require_available_or_raise(1)
    try:
        r = requests.get(ETHERSCAN_BASE_URL, params=params, timeout=ETHERSCAN_HTTP_TIMEOUT)
        if r.status_code == 429:
            # Treat 429 as exhausted for the day (conservative)
            raise QuotaExceeded("HTTP 429 Too Many Requests")
        r.raise_for_status()
        return r.json()
    except QuotaExceeded:
        raise
    except Exception as e:
        logger.exception(f"HTTP request failed: {e}")
        raise


def get_most_recent_blocknumber(chainid: int = 1) -> int:
    params = {"chainid": chainid, "module": "proxy", "action": "eth_blockNumber", "apikey": ETHERSCAN_API_KEY}
    data = _get(params)
    return int(data["result"], 16)


def get_events_by_contract_addr(addr: str, chainid: int = 1):
    params = {"chainid": chainid, "module": "logs", "action": "getLogs", "apikey": ETHERSCAN_API_KEY, "address": addr}
    data = _get(params)
    status = data.get("status")
    if status == "1":
        return data.get("result", [])
    if "result" in data and isinstance(data["result"], list):
        return data["result"]
    raise Exception(data.get("message", "Unknown error"))


def get_normal_transactions_by_contract_addr(addr: str, chainid: int = 1):
    params = {
        "chainid": chainid,
        "module": "account",
        "action": "txlist",
        "address": addr,
        "startblock": 0,
        "endblock": get_most_recent_blocknumber(chainid),
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY,
    }
    data = _get(params)
    if data.get("status") == "1":
        return data.get("result", [])
    if isinstance(data.get("result"), list):
        return data["result"]
    raise Exception(data.get("message", "Unknown error"))


def get_internal_transactions_by_contract_addr(addr: str, chainid: int = 1):
    params = {
        "chainid": chainid,
        "module": "account",
        "action": "txlistinternal",
        "address": addr,
        "startblock": 0,
        "endblock": get_most_recent_blocknumber(chainid),
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY,
    }
    data = _get(params)
    if data.get("status") == "1":
        return data.get("result", [])
    if isinstance(data.get("result"), list):
        return data["result"]
    raise Exception(data.get("message", "Unknown error"))


def get_contract_creator_by_contract_addr(addr: str, chainid: int = 1):
    params = {
        "chainid": chainid,
        "module": "contract",
        "action": "getcontractcreation",
        "contractaddresses": addr,
        "apikey": ETHERSCAN_API_KEY,
    }
    data = _get(params)
    if data.get("status") == "1":
        res = data.get("result", [])
        return res[0] if isinstance(res, list) and res else {}
    res = data.get("result")
    if isinstance(res, list) and res:
        return res[0]
    raise Exception(data.get("message", "Unknown error"))


def get_balance_by_contract_addr(addr: str, chainid: int = 1) -> int:
    params = {
        "chainid": chainid,
        "module": "account",
        "action": "balance",
        "address": addr,
        "tag": "latest",
        "apikey": ETHERSCAN_API_KEY,
    }
    data = _get(params)
    if data.get("status") == "1" and "result" in data:
        return int(data["result"])
    if "result" in data:
        return int(data["result"])
    raise Exception(data.get("message", "Unknown error"))


def get_token_supply_by_contract_addr(addr: str, chainid: int = 1) -> int:
    params = {"chainid": chainid, "module": "stats", "action": "tokensupply", "contractaddress": addr, "apikey": ETHERSCAN_API_KEY}
    data = _get(params)
    if data.get("status") == "1" and "result" in data:
        return int(data["result"])
    if "result" in data:
        return int(data["result"])
    raise Exception(data.get("message", "Unknown error"))


def get_transaction_count_by_contract_addr(addr: str, chainid: int = 1) -> int:
    params = {"chainid": chainid, "module": "proxy", "action": "eth_getTransactionCount", "address": addr, "tag": "latest", "apikey": ETHERSCAN_API_KEY}
    data = _get(params)
    if "result" in data:
        return int(data["result"], 16)  # hex to int
    raise Exception(data.get("message", "Unknown error"))


def get_info_by_contract_addr(addr: str, chainid: int = 1) -> dict:
    """
    Composite: creator, balance, normal+internal tx, events.
    Underlying calls already check the quota individually.
    """
    info = dict()
    info["creator"] = get_contract_creator_by_contract_addr(addr, chainid)
    info["balance"] = get_balance_by_contract_addr(addr, chainid)
    info["transaction"] = get_normal_transactions_by_contract_addr(addr, chainid) + get_internal_transactions_by_contract_addr(addr, chainid)
    info["event"] = get_events_by_contract_addr(addr, chainid)
    time.sleep(0.2)
    return info


def get_bytecode_by_contract_addr(addr: str, chainid: int = 1) -> str:
    params = {"chainid": chainid, "module": "proxy", "action": "eth_getCode", "address": addr, "tag": "latest", "apikey": ETHERSCAN_API_KEY}
    data = _get(params)
    if "result" in data:
        return data["result"]
    raise Exception(data.get("message", "Unknown error"))


def get_source_code_by_contract_addr(addr: str, chainid: int = 1):
    params = {"chainid": chainid, "module": "contract", "action": "getsourcecode", "address": addr, "apikey": ETHERSCAN_API_KEY}
    data = _get(params)
    if data.get("status") == "1":
        res = data.get("result", [])
        return res[0] if isinstance(res, list) and res else {}
    res = data.get("result")
    if isinstance(res, list) and res:
        return res[0]
    raise Exception(data.get("message", "Unknown error"))


def reached_limit() -> dict:
    """
    Returns raw getapilimit result for debugging/telemetry.
    Prefer using EtherscanQuota for enforcement.
    """
    params = {"module": "getapilimit", "action": "getapilimit", "apikey": ETHERSCAN_API_KEY}
    try:
        data = _get(params)
        return data.get("result", {})
    except QuotaExceeded:
        return {"dailyRemaining": 0}
    except Exception as e:
        logger.warning(f"reached_limit fallback: {e}")
        return {}


# -------------------------------
# File save helpers (RESTORED)
# -------------------------------

def save_bytecode_by_contract_addr(save_folder, addr, bytecode):
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{addr.lower()}.hex"
    file_path = os.path.join(save_folder, filename)
    with open(file_path, "w") as f:
        f.write(bytecode)
    logger.info(f"Saved {filename}")
    return file_path


def save_transactions_by_contract_addr(save_folder, addr, info):
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{addr.lower()}.json"
    file_path = os.path.join(save_folder, filename)
    with open(file_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info(f"Saved {filename}")
    return file_path


def save_sol_by_contract_addr(save_folder, addr, source):
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{addr.lower()}.sol"
    file_path = os.path.join(save_folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(source)
    logger.info(f"Saved {filename}")
    return file_path
