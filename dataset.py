import os
import json
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from config import device, DEBUG, USER_FEATURES_PATH, FMA_FEATURES_PATH, USER_INDEX, COMBINED_INDEX
from collections import Counter
import math
import h5py
from collections import defaultdict, Counter
        
MAX_TIME_STEPS = 2401
STEP_STRIDE = 2401  


# In dataset.py, modify GenreDataset class
class GenreDataset(Dataset):
    def __init__(self, features_hdf5_path, genre_to_index):
        self.genre_to_index = genre_to_index
        self.features_hdf5_path = features_hdf5_path
        self.segments = []
        
        

        TOP_N_GENRES = 20
        MAX_PER_GENRE = 100
       

        # First pass: count all genre occurrences
        genre_counter = Counter()

        with h5py.File(features_hdf5_path, "r") as hdf5_file:
            for file_name in hdf5_file:
                if "genre" in hdf5_file[file_name].attrs:
                    genre_raw = hdf5_file[file_name].attrs["genre"]
                    genre_list = [g.strip() for g in genre_raw.split(",")] if isinstance(genre_raw, str) else [genre_raw]
                    genre_counter.update(genre_list)

        # Get top N genres
        top_genres = set([genre for genre, _ in genre_counter.most_common(TOP_N_GENRES)])
        print(f"✅ Top {TOP_N_GENRES} genres: {top_genres}")

        # Second pass: filter and balance
        genre_counts = defaultdict(int)
        filtered_segments = []

        with h5py.File(features_hdf5_path, "r") as hdf5_file:
            for file_name in hdf5_file:
                if "genre" in hdf5_file[file_name].attrs:
                    total_steps = hdf5_file[file_name].shape[1]
                    for start in range(0, total_steps, STEP_STRIDE):
                        genre_raw = hdf5_file[file_name].attrs["genre"]
                        genre_list = [g.strip() for g in genre_raw.split(",")] if isinstance(genre_raw, str) else [genre_raw]

                        # Only keep if all genres are in top genres
                        if all(g in top_genres for g in genre_list):
                            if all(genre_counts[g] < MAX_PER_GENRE for g in genre_list):
                                filtered_segments.append((file_name, start, genre_list))
                                for g in genre_list:
                                    genre_counts[g] += 1

        print(f"✅ Final filtered segments: {len(filtered_segments)}")

        
    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        file_name, start, genre_list = self.segments[idx]
        
        # Open file for just this read
        with h5py.File(self.features_hdf5_path, "r", swmr=True) as hdf5_file:
            features = np.array(hdf5_file[file_name])
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

            end = start + MAX_TIME_STEPS
            segment = features[:, start:end]

            if segment.shape[1] < MAX_TIME_STEPS:
                pad_width = ((0, 0), (0, MAX_TIME_STEPS - segment.shape[1]))
                segment = np.pad(segment, pad_width, mode="constant")

        multi_hot = torch.zeros(len(self.genre_to_index), dtype=torch.float32)
        for genre in genre_list:
            g_idx = self.genre_to_index.get(genre)
            if g_idx is not None:
                multi_hot[g_idx] = 1.0
            
        return torch.tensor(segment, dtype=torch.float32), multi_hot


def load_datasets():
    with open(USER_INDEX) as f:
        user_index = json.load(f)
    with open(COMBINED_INDEX, "r") as f:
        combined_index = json.load(f)

    user_dataset = GenreDataset(USER_FEATURES_PATH, combined_index)
    fma_dataset = GenreDataset(FMA_FEATURES_PATH, combined_index)

    return user_dataset, fma_dataset, user_index, combined_index



from collections import Counter
import math

def compute_class_weights(h5_paths,genre_to_index, max_clip=5.0, apply_log=True):
    

    total_counts = Counter()

    for path in h5_paths:
        with h5py.File(path, "r") as f:
            for song_id in f:
                if "genre" in f[song_id].attrs:
                    raw = f[song_id].attrs["genre"]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    genres = [g.strip() for g in raw.split(",")]

                    # 🔥 Count number of segments (not just songs)
                    total_steps = f[song_id].shape[1]
                    segment_count = max(1, total_steps // STEP_STRIDE)

                    for g in genres:
                        total_counts[g] += segment_count

    weights = np.ones(len(genre_to_index), dtype=np.float32)

    for genre, count in total_counts.items():
        if genre in genre_to_index and count > 0:
            weight = 1.0 / math.log(1 + count) if apply_log else 1.0 / count
            weight = min(weight, max_clip)
            weights[genre_to_index[genre]] = weight

    weights /= np.mean(weights)  # Normalize to mean = 1
    print(f"📊 Class Weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")

    return weights

def load_genre_index(path):
    with open(path, "r") as f:
        genre_to_index = json.load(f)
    index_to_genre = {i: g for g, i in genre_to_index.items()}
    return genre_to_index, index_to_genre
