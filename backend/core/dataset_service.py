import pandas as pd
from tqdm import tqdm
from typing import List, Optional

from backend.utils.constants import LABELED_PATH, FEATURE_PATH
from backend.core.feature_service import extract_base_feature_from_address

def get_full_dataset(filename=LABELED_PATH / 'groundtruth.csv', refresh=False, addresses:Optional[List[str]]=None):
    df = pd.read_csv(filename, index_col=0)

    if addresses:
        df = df[df.index.isin(addresses)]

    keys = df.index.tolist()
    dataset = dict()

    for address in tqdm(keys):
        feature = extract_base_feature_from_address(address, output_dir=FEATURE_PATH, refresh=refresh)
        dataset[address] = {
          **feature,
          "Label": df.loc[address].to_dict()
        }

    return dataset
