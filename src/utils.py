import random
import numpy as np

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())
