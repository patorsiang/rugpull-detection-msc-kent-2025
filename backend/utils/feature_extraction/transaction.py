import json
import pandas as pd
from collections import Counter
from pathlib import Path
import requests
import redis
import numpy as np
import os
# === Global Cache (Redis) ===
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def lookup_4byte(topic_hash):
    """Lookup event signature using 4byte.directory API with Redis caching."""
    cached = r.get(topic_hash)
    if cached:
        return cached

    url = f"https://www.4byte.directory/api/v1/event-signatures/?hex_signature={topic_hash}"
    try:
        response = requests.get(url, timeout=5)
        results = response.json().get("results", [])
        if results:
            decoded = results[0].get('text_signature', 'unknown')
            r.set(topic_hash, decoded)
            return decoded
    except Exception as e:
        print(f"⚠️ Error decoding {topic_hash}: {e}")

    r.set(topic_hash, "unknown")
    return "unknown"

def load_transaction(txn_file):
    with open(txn_file) as f:
        data = json.load(f)

    address = Path(txn_file).stem
    transactions = data.get("transaction", [])
    events = data.get("event", [])
    creator = data.get("creator", {})
    return address, creator, transactions, events

# Safe statistical functions
def safe_mean(x): return float(np.mean(x)) if x else 0
def safe_max(x): return max(x) if x else 0
def safe_min(x): return min(x) if x else 0
def safe_std(x): return float(np.std(x)) if x else 0

def extract_transaction_features(txn_file):
    address, creator, transactions, events  = load_transaction(txn_file)

    creator_address = creator.get("contractAddress", "").lower()

    # Count stats
    txn_nums = len(transactions)
    event_nums = len(events)

    # Creation
    creation_block = int(creator.get('blockNumber', 0))
    creation_timestamp = int(creator.get('timestamp', 0))

    # Trackers
    block_numbers = []
    timestamps = []
    topic_list = []
    function_list = []
    gas_limits = []
    gas_used_list = []
    gas_price_list = []
    all_addresses = set()
    buyers = set()
    sellers = set()
    num_creator_transfers = 0
    buy_amt = 0
    sell_amt = 0

    # Event signature collection
    for event in events:
        topic_list += event.get('topics', [])

    topic_counter = Counter(topic_list)
    event_topic_counter = dict()
    for type_key, num in topic_counter.items():
        decoded = lookup_4byte(type_key)
        if decoded != 'unknown':
            event_topic_counter[f"{decoded.split('(')[0]}_num".lower()] = num

    # Transaction-level analysis
    for txn in transactions:
        block_number = np.float64(txn.get('blockNumber', 0))
        timestamp = np.float64(txn.get('timeStamp', 0))
        to_addr = txn.get('to', '').lower()
        from_addr = txn.get('from', '').lower()
        value = np.float64(txn.get("value", 0))
        function_name = txn.get('functionName', "").split('(')[0]
        gas_limits.append(np.float64(txn.get('gas', 0)))
        gas_used_list.append(np.float64(txn.get('gasUsed', 0)))
        gas_price_list.append(np.float64(txn.get('gasPrice', 0)))

        if block_number:
            block_numbers.append(block_number)
        if timestamp:
            timestamps.append(timestamp)

        all_addresses.update([to_addr, from_addr])

        if to_addr != creator_address and from_addr != to_addr:
            buyers.add(to_addr)
        if from_addr == creator_address:
            num_creator_transfers += 1
            buy_amt += value
        else:
            sell_amt += value
        if from_addr != to_addr:
            sellers.add(from_addr)

        if function_name:
            function_list.append(f"{function_name}_num".lower())

    # Function signature stats
    function_counter = Counter(function_list)
    # Time features
    start_block = min(block_numbers) if block_numbers else 0
    end_block = max(block_numbers) if block_numbers else 0
    start_ts = min(timestamps) if timestamps else 0
    end_ts = max(timestamps) if timestamps else 0
    life_time = end_block - start_block
    duration_seconds = end_ts - start_ts

    # Feature summary
    features = {
        "Address": address,
        "txn_nums": txn_nums,
        "event_nums": event_nums,
        "creation_block": creation_block,
        "creation_timestamp": creation_timestamp,
        "start_block": start_block,
        "end_block": end_block,
        "life_time": life_time,
        "start_time": start_ts,
        "end_time": end_ts,
        "duration_seconds": duration_seconds,
        "from_creation_to_transfer": start_ts - creation_timestamp if creation_timestamp > 0 else 0,
        "num_addresses": len(all_addresses),
        "num_buyers": len(buyers),
        "num_sellers": len(sellers),
        "num_creator_transfers": num_creator_transfers,
        "buy_amt": buy_amt,
        "sell_amt": sell_amt,
        "avg_value": (buy_amt + sell_amt) / (txn_nums + 1),
        "txn_per_block": txn_nums / (life_time + 1) if life_time > 0 else 0,
        "avg_gas_limit": safe_mean(gas_limits),
        "max_gas_limit": safe_max(gas_limits),
        "min_gas_limit": safe_min(gas_limits),
        "std_gas_limit": safe_std(gas_limits),
        "avg_gas_used": safe_mean(gas_used_list),
        "max_gas_used": safe_max(gas_used_list),
        "min_gas_used": safe_min(gas_used_list),
        "std_gas_used": safe_std(gas_used_list),
        "avg_gas_price": safe_mean(gas_price_list),
        "max_gas_price": safe_max(gas_price_list),
        "min_gas_price": safe_min(gas_price_list),
        "std_gas_price": safe_std(gas_price_list),
    }

    # Merge with dynamic event/function features
    features.update(event_topic_counter)
    features.update(function_counter)

    return features

def save_txn_feature_dataframe(out_dir):
    """Process all JSON files in a folder and build a feature DataFrame."""
    feature_rows = []
    for txn_file in Path(os.path.join(out_dir, 'txn')).glob("*.json"):
        try:
            row = extract_transaction_features(txn_file)
            feature_rows.append(row)
        except Exception as e:
            print(f"Failed to process {txn_file}: {e}")

    df = pd.DataFrame(feature_rows)
    df = df.set_index("Address")
    df = df.fillna(0)
    df.to_csv(os.path.join(out_dir, f"txn_features.csv"))
    return df
