import pandas as pd
from tqdm import tqdm
from typing import List, Optional
from backend.utils.constants import DATA_PATH, FEATURE_PATH
from backend.core.feature_service import extract_base_feature_from_address
from backend.utils.logger import logging
from backend.utils.etherscan_quota import quota_guard, QuotaExceeded

logger = logging.getLogger(__name__)

def get_full_dataset(
    filename: str = "groundtruth.csv",
    refresh: bool = False,
    addresses: Optional[List[str]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    abort_if_quota_exhausted: bool = True,
):
    """
    Build a dict: address -> { feature_k: v, ..., "Label": {label: 0/1 or -1} }
    - addresses: restrict to these (if provided)
    - limit/offset: simple window for large files
    - refresh: re-extract even if cached features exist
    - abort_if_quota_exhausted: if True, and features are missing + quota is 0 -> abort early
    """
    path = DATA_PATH / filename
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.lower()

    if addresses:
        addresses = [a.lower() for a in addresses]
        df = df[df.index.isin(addresses)]

    if offset:
        df = df.iloc[offset:]
    if limit:
        df = df.iloc[:limit]

    keys = df.index.tolist()

    # Pre-scan for missing feature caches
    if not refresh and abort_if_quota_exhausted:
        missing = [a for a in keys if not (FEATURE_PATH / f"{a}.json").exists()]
        if missing and quota_guard.is_exhausted(require_calls=1):
            raise QuotaExceeded(
                f"Etherscan quota exhausted and {len(missing)} addresses need fresh features. Aborting."
            )

    dataset = {}
    logger.info(f"[get_full_dataset] loading {len(keys)} addresses from {filename} (refresh={refresh})")
    for address in tqdm(keys):
        try:
            feature = extract_base_feature_from_address(address, output_dir=str(FEATURE_PATH), refresh=refresh)
        except QuotaExceeded as e:
            logger.warning(f"quota hits while extracting {address}: {e}")
            raise
        except Exception as e:
            logger.exception(f"feature extraction failed for {address}: {e}")
            feature = {"timeline_sequence": [], "sourcecode": "", "opcode_sequence": ""}

        labels = df.loc[address].to_dict()
        dataset[address] = {**feature, "Label": labels}

    return dataset
