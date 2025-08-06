from pathlib import Path
from backend.utils.download import download_contract_from_etherscan

def prepare_contract_download(address: str):
    """
    Downloads contract data (if needed) and returns file Paths for hex, sol, and txn.
    """
    files = download_contract_from_etherscan(address)

    if not files:
        return None

    # Convert to Path objects and only return ones that actually exist
    return [Path(f) for f in files if f and Path(f).exists()]
