import os
import json
import requests

# def get_transactions_for_contract(api_key, addr, max_txn=1000):
def get_transactions_for_contract(api_key, addr, max_pages=10):
    url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
    headers = {"Content-Type": "application/json"}

    method = "alchemy_getAssetTransfers"
    all_transfers = []
    page_key = None
    pages_fetched = 0
    seen_ids = set()

    while not page_key or pages_fetched < max_pages:
        body = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": method,
            "params": [{
                "fromBlock": "0x0",
                "toBlock": "latest",
                "contractAddresses": [addr],
                "withMetadata": True,
                "category": ["external", "internal", "erc20", "erc721", "erc1155", "specialnft"],
                "order": "asc"
            }]
        }

        if page_key:
            body["params"][0]["pageKey"] = page_key

        response = requests.post(url, headers=headers, json=body).json()
        result = response.get("result", {})
        page_key = result.get("pageKey", None)

        new_transfers = result.get("transfers", [])
        deduped = deduplicate_txns(new_transfers, seen_ids)

        if not deduped:
            print(f"🚫 Page {pages_fetched + 1} returned only duplicates. Stopping early.")
            break  # ✅ DO THIS INSTEAD OF RETURNING NOTHING

        all_transfers.extend(deduped)
        pages_fetched += 1

    print(f"✅ Fetched {len(all_transfers)} unique transfers for {addr}")
    return all_transfers


def save_transactions_for_contract(save_folder, addr, txns):
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{addr.lower()}.json"
    file_path = os.path.join(save_folder, filename)

    with open(file_path, "w") as f:
        json.dump(txns, f, indent=2)

    return file_path

def deduplicate_txns(transfers, seen_ids):
    deduped = []
    for tx in transfers:
        uid = tx.get("uniqueId") or tx.get("hash")
        if uid and uid not in seen_ids:
            seen_ids.add(uid)
            deduped.append(tx)
    return deduped
