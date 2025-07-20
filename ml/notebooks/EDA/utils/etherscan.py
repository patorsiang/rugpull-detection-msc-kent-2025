import warnings
warnings.filterwarnings("ignore")

import os
import requests
from dotenv import load_dotenv
from pathlib import Path


URL = "https://api.etherscan.io/v2/api"
project_root = Path.cwd().parents[3]
load_dotenv(os.path.join(project_root, ".env"))
api_key = os.getenv("ETHERSCAN_API_KEY")

def get_source_code(address, chain_id=1):
  params = {
    "chainid": chain_id,
    "module": "contract",
    "action": "getsourcecode",
    "address": address,
    "apikey": api_key
  }
  try:
    res = requests.get(URL, params=params).json()
    return res['result'][0] if res['status'] == '1' else None
  except Exception:
    return None

def get_bytecode(address, chain_id=1):
  params = {
    "chainid": chain_id,
    "module": "proxy",
    "action": "eth_getCode",
    "address": address,
    "tag": "latest",
    "apikey": api_key
  }
  try:
    res = requests.get(URL, params=params).json()
    return None if res['result'] == '0x' else res['result']
  except Exception:
    return None

def save_code(mode, address, code, saving_dir, chain_id=1):
    extension = 'sol' if mode == 'source' else 'hex'
    filename = os.path.join(saving_dir, f"{address}.{extension}")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Saved {mode} for {address} (chain {chain_id})")

