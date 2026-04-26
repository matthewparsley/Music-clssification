import os
import re
import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from config import SAMPLE_RATE, DEBUG, USER_AUDIO_DIR, GENRE_CORRECTIONS


def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)
	

def variable_collate_fn(batch):
    """Collate function to handle variable-sized feature vectors in a batch."""
    features, labels = zip(*batch)  # ✅ Unpack features and labels

    # ✅ Convert features to tensors (keep variable sizes)
    feature_tensors = [torch.tensor(f, dtype=torch.float32) for f in features]
    label_tensors = torch.stack(labels).float()  # instead of long/int


    # ✅ Pad features to match the longest feature in the batch
    padded_features = pad_sequence(feature_tensors, batch_first=True, padding_value=0)

    return padded_features, label_tensors


