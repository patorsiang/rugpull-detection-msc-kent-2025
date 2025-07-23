import os
import json
import requests

def get_transactions_by_contract_addr(api_key, addr):
    url = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "txlist",
        "address": addr,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": api_key
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "1":
            return data["result"]
        else:
            print(f"❌ No transactions found or error: {data.get('message', '')}")
            return []
    except Exception as e:
        print(f"🚨 Error fetching transactions: {e}")
        return []



def save_transactions_by_contract_addr(save_folder, addr, txns):
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{addr.lower()}.json"
    file_path = os.path.join(save_folder, filename)

    with open(file_path, "w") as f:
        json.dump(txns, f, indent=2)

    return file_path

# def get_bytecode_by_contract_addr(api_key, addr):
#     url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
#     payload = {
#         "jsonrpc": "2.0",
#         "id": 1,
#         "method": "eth_getCode",
#         "params": [addr, "latest"]
#     }
#     headers = {"Content-Type": "application/json"}

#     try:
#         response = requests.post(url, json=payload, headers=headers)
#         data = response.json()
#         if "result" in data:
#             return data["result"]
#         else:
#             print(f"❌ Error from Alchemy: {data.get('error', {}).get('message', 'Unknown error')}")
#             return ""
#     except Exception as e:
#         print(f"🚨 Exception fetching bytecode: {e}")
#         return ""

def get_bytecode_by_contract_addr(api_key, addr):
    url = "https://api.etherscan.io/v2/api"
    params = {
        "chainid": 1,  # Ethereum mainnet
        "module": "proxy",
        "action": "eth_getCode",
        "address": addr,
        "tag": "latest",
        "apikey": api_key
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "result" in data:
            return data["result"]
        else:
            print(f"❌ Etherscan Error: {data.get('message', 'Unknown error')}")
            return ""
    except Exception as e:
        print(f"🚨 Exception fetching bytecode from Etherscan: {e}")
        return ""


def save_bytecode_by_contract_addr(save_folder, addr, bytecode):
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{addr.lower()}.hex"
    file_path = os.path.join(save_folder, filename)

    with open(file_path, 'w') as f:
        f.write(bytecode)

    return file_path
