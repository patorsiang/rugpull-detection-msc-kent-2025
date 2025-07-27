import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

url = "https://api.etherscan.io/v2/api"

def get_most_recent_blocknumber(chainid=1):
    params = {
        "chainid": chainid,  # Ethereum mainnet = 1
        "module": "proxy",
        "action": "eth_blockNumber",
        "apikey": ETHERSCAN_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        return int(data["result"], 16)
    except Exception as e:
        print(f"error: {e}")
        return 0

def get_events_by_contract_addr(addr, chainid=1):
    params = {
        "chainid": chainid,  # Ethereum mainnet = 1
        "module": "logs",
        "action": "getLogs",
        "apikey": ETHERSCAN_API_KEY,
        "address": addr,
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "1":
            return data["result"]
        else:
            print(f"error: {data.get('message', '')}")
            return []
    except Exception as e:
        print(f"error: {e}")
        return []

def get_normal_transactions_by_contract_addr(addr, chainid=1):
    params = {
        "chainid": chainid,  # Ethereum mainnet = 1
        "module": "account",
        "action": "txlist",
        "address": addr,
        "startblock": 0,
        "endblock": get_most_recent_blocknumber(chainid),
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "1":
            return data["result"]
        else:
            print(f"error: {data.get('message', '')}")
            return []
    except Exception as e:
        print(f"error: {e}")
        return []

def get_internal_transactions_by_contract_addr(addr, chainid=1):
    params = {
        "chainid": chainid,  # Ethereum mainnet = 1
        "module": "account",
        "action": "txlistinternal",
        "address": addr,
        "startblock": 0,
        "endblock": get_most_recent_blocknumber(chainid),
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "1":
            return data["result"]
        else:
            print(f"error: {data.get('message', '')}")
            return []
    except Exception as e:
        print(f"error: {e}")
        return []

def get_contract_creator_by_contract_addr(addr, chainid=1):
    params = {
        "chainid": chainid,  # Ethereum mainnet = 1
        "module": "contract",
        "action": "getcontractcreation",
        "contractaddresses": addr,
        "apikey": ETHERSCAN_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "1":
            return data["result"][0]
        else:
            print(f"error: {data.get('message', '')}")
            return dict()
    except Exception as e:
        print(f"error: {e}")
        return dict()

def get_balance_by_contract_addr(addr, chainid=1):
    params = {
        "chainid": chainid,  # Ethereum mainnet = 1
        "module": "account",
        "action": "balance",
        "address": addr,
        "tag": "latest",
        "apikey": ETHERSCAN_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "1":
            return int(data["result"])
        else:
            print(f"error: {data.get('message', '')}")
            return 0
    except Exception as e:
        print(f"error: {e}")
        return 0

def get_token_supply_by_contract_addr(addr, chainid=1):
    params = {
        "chainid": chainid,  # Ethereum mainnet = 1
        "module": "stats",
        "action": "tokensupply",
        "contractaddress": addr,
        "apikey": ETHERSCAN_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "1":
            return int(data["result"])
        else:
            print(f"error: {data.get('message', '')}")
            return 0
    except Exception as e:
        print(f"error: {e}")
        return 0

def get_transaction_count_by_contract_addr(addr, chainid=1):
    params = {
        "chainid": chainid,  # Ethereum mainnet = 1
        "module": "proxy",
        "action": "eth_getTransactionCount",
        "address": addr,
        "tag": "latest",
        "apikey": ETHERSCAN_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "result" in data:
            return int(data["result"], 16)

        return 0
    except Exception as e:
        print(f"error: {e}")
        return 0

def get_transaction_count_by_contract_addr(addr, chainid=1):
    params = {
        "chainid": chainid,  # Ethereum mainnet = 1
        "module": "proxy",
        "action": "eth_getTransactionCount",
        "address": addr,
        "tag": "latest",
        "apikey": ETHERSCAN_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "1":
            return int(data["result"])
        else:
            print(f"error: {data.get('message', '')}")
            return 0
    except Exception as e:
        print(f"error: {e}")
        return 0

def get_info_by_contract_addr(addr, chainid=1):
    info = dict()
    info['creator'] = get_contract_creator_by_contract_addr(addr, chainid)
    info['balance'] = get_balance_by_contract_addr(addr, chainid)
    info['transaction'] = get_normal_transactions_by_contract_addr(addr, chainid) + get_internal_transactions_by_contract_addr(addr, chainid)
    info['event'] = get_events_by_contract_addr(addr, chainid)
    time.sleep(1)
    return info

def get_bytecode_by_contract_addr(addr, chainid=1):
    params = {
        "chainid": chainid,  # Ethereum mainnet = 1
        "module": "proxy",
        "action": "eth_getCode",
        "address": addr,
        "tag": "latest",
        "apikey": ETHERSCAN_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "result" in data:
            return data["result"]
        else:
            print(f"error: {data.get('message', 'Unknown error')}")
            return ""
    except Exception as e:
        print(f"error: {e}")
        return ""

def get_source_code_by_contract_addr(addr, chainid=1):
    params = {
        "chainid": chainid,  # Ethereum mainnet = 1
        "module": "contract",
        "action": "getsourcecode",
        "contractaddresses": addr,
        "apikey": ETHERSCAN_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "1":
            return data["result"][0]
        else:
            print(f"error: {data.get('message', '')}")
            return dict()
    except Exception as e:
        print(f"error: {e}")
        return dict()

def reached_limit():
    params = {
        "module": "getapilimit",
        "action": "getapilimit",
        "apikey": ETHERSCAN_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "1":
            return data["result"]
        else:
            print(f"error: {data.get('message', '')}")
            if "result" in data:
                return data["result"]
            return dict()
    except Exception as e:
        print(f"error: {e}")
        return dict()

def save_bytecode_by_contract_addr(save_folder, addr, bytecode):
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{addr.lower()}.hex"
    file_path = os.path.join(save_folder, filename)

    with open(file_path, 'w') as f:
        f.write(bytecode)

    return file_path

def save_transactions_by_contract_addr(save_folder, addr, info):
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{addr.lower()}.json"
    file_path = os.path.join(save_folder, filename)
    with open(file_path, "w") as f:
        json.dump(info, f, indent=4)

    return file_path

def save_sol_by_contract_addr(save_folder, addr, source):
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{addr.lower()}.sol"
    file_path = os.path.join(save_folder, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(source)

    return file_path
