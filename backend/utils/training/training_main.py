from sklearn.model_selection import train_test_split
from backend.core.dataset_service import get_full_dataset

def train_and_save_best_model(test_size=0.2):
    dataset = get_full_dataset()
