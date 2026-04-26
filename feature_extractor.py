import os
import json
import numpy as np
import torch
import torchaudio
import librosa
import h5py
from tqdm import tqdm
import torch.nn.functional as F
import multiprocessing
from torch.multiprocessing import Pool
import fasteners 
from config import USER_AUDIO_DIR, USER_FEATURES_PATH, DEBUG,  FMA_AUDIO_DIR, FMA_FEATURES_PATH, USER_GENRE_FILE, FMA_GENRE_FILE
from time import time
from librosa.feature.rhythm import tempo as compute_tempo

def load_genre_mapping(path):
    with open(path, "r") as f:
        return json.load(f)

USER_GENRES = load_genre_mapping(USER_GENRE_FILE)
FMA_GENRES = load_genre_mapping(FMA_GENRE_FILE)



def match_time_steps(tensor, target_size):
    """Resample a tensor to match target time steps using nearest neighbor interpolation."""
    if tensor.shape[1] == target_size:
        return tensor  # Already the correct shape
    return F.interpolate(tensor.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)

def pad_to_target(tensor, target_dim):
    """Pads tensor along dim=0 to match target_dim."""
    current_dim = tensor.shape[0]
    if current_dim < target_dim:
        pad_size = target_dim - current_dim
        padding = torch.zeros((pad_size, tensor.shape[1]), dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding], dim=0)  # Pad along the first dimension
    return tensor

FAILED_LOG_FILE = "failed_extractions.txt"

def log_failed_extraction(audio_path, error_message):
    """Log songs that failed feature extraction."""
    with open(FAILED_LOG_FILE, "a") as f:
        f.write(f"{audio_path} - ERROR: {error_message}\n")


def extract_features(audio_path):
    try:
   
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)

        # ✅ If sample rate is not 16kHz, resample
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        sample_rate = 16000

        waveform = waveform.to("cpu")

        # ✅ Convert stereo to mono
        if waveform.shape[0] > 1:
     
            waveform = waveform.mean(dim=0)
        
        waveform_np = waveform.numpy().squeeze()

        mel_spec = librosa.feature.melspectrogram(y=waveform_np, sr=16000, n_mels=64, n_fft=400, hop_length=200)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)


        mfcc = librosa.feature.mfcc(y=waveform_np, sr=16000, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_tensor = torch.tensor(np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0), dtype=torch.float32)

        additional_features = extract_additional_features(waveform_np, 16000, max_time_steps=mel_spec.shape[1])
 
        bpm = compute_bpm_from_waveform(waveform_np, 16000)
        bpm_feature = torch.tensor(bpm, dtype=torch.float32).view(1, 1).expand(64, mel_spec.shape[1])

        max_time_steps = mel_spec.shape[1]
        target_dim = mel_spec.shape[0]

        mfcc_tensor = match_time_steps(mfcc_tensor, max_time_steps)
        additional_features = match_time_steps(additional_features, max_time_steps)
 
        mfcc_tensor = pad_to_target(mfcc_tensor, target_dim)
        additional_features = pad_to_target(additional_features, target_dim)
 
        features = torch.cat([bpm_feature, mel_spec, mfcc_tensor, additional_features], dim=0)

        mean = features.mean(dim=1, keepdim=True)  # Compute per-feature mean
        std = features.std(dim=1, keepdim=True)    # Compute per-feature std
        std[std == 0] = 1  # Prevent division by zero

        features = (features - mean) / std  # Standardization

    except Exception as e:
        error_message = f"❌ ERROR: Feature extraction failed for {audio_path}: {e}"
        print(error_message)  # ✅ Show error in terminal
        log_failed_extraction(audio_path, str(e))  # ✅ Log error to file

            # ✅ Return a zero placeholder tensor to prevent dataset corruption
        max_time_steps = 500  # Set a default safe value
        return torch.zeros((256, max_time_steps), dtype=torch.float32)
    return features



def extract_additional_features(y, sr, max_time_steps):
    """Extract Spectral Features: Chroma, Spectral Contrast, Roll-off, ZCR, Tonnetz, RMS."""
    try:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # Shape: (1, TimeSteps)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  # Shape: (1, TimeSteps)
        zcr = librosa.feature.zero_crossing_rate(y)  # Shape: (1, TimeSteps)
        rms = librosa.feature.rms(y=y)  # Shape: (1, TimeSteps)

        # Convert to Tensors
        features_list = [centroid, rolloff, zcr, rms]
        features_list = [torch.tensor(f, dtype=torch.float32) for f in features_list]

        # Ensure max_time_steps is valid before padding
        if max_time_steps is None or max_time_steps <= 0:
            max_time_steps = min(f.shape[1] for f in features_list if f is not None and f.shape[1] > 0)

        # Apply padding safely
        features_list = [
            torch.nn.functional.pad(f, (0, max_time_steps - f.shape[1]))[:, :max_time_steps]
            if f.shape[1] < max_time_steps else f[:, :max_time_steps] 
            for f in features_list
        ]

        # Stack All Features
        additional_features = torch.cat(features_list, dim=0)
        if DEBUG:
            print(f"✅ DEBUG: Additional Features Stacking Successful - Shape: {additional_features.shape}")
        return additional_features

    except Exception as e:
        print(f"⚠️ WARNING: Failed to extract additional features: {e}")
        return torch.zeros((20, max_time_steps if max_time_steps else 100))  # Safe fallback



def compute_bpm_from_waveform(y, sr):
    """Compute BPM from onset envelope using safe fallback logic."""
    try:
        if y is None or len(y) == 0:
            print("⚠️ Empty waveform for BPM")
            return 0

        if np.isnan(y).any():
            print("⚠️ NaNs in waveform")
            return 0

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if onset_env is None or len(onset_env) == 0:
            print("⚠️ Onset envelope is empty")
            return 0

        tempos = compute_tempo(onset_envelope=onset_env, sr=sr)
        if tempos is not None and len(tempos) > 0:
            return float(tempos[0])
        else:
            print("⚠️ tempo() returned empty or None")
            return 0
    except Exception as e:
        print(f"⚠️ WARNING: Failed to compute BPM: {e}")
        return 0

def get_all_audio_files(directory):
    """Recursively find all MP3 files in the given directory and its subfolders."""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".mp3")):
                                audio_files.append(os.path.join(root, file))  # ✅ Full path to file
    return audio_files

def process_audio_wrapper(args):
    """Wrapper function for multiprocessing."""
    return process_audio_file(*args)

def process_audio_file(file_path, hdf5_path):
    
    file_name = os.path.basename(file_path)

    try:
        features = extract_features(file_path)

        file_name = os.path.basename(file_path)

        # Detect whether it's an FMA or User track
        genre_source = FMA_GENRES if "fma_medium" in file_path else USER_GENRES
        genres = genre_source.get(file_name)

        # If no genres found, skip this file
        if not genres:
            print(f"⚠️ No genre found for {file_name}, skipping.")
            return None

        # ✅ Correct HDF5 locking
        lock = fasteners.InterProcessLock(hdf5_path + ".lock")
        try:
            lock.acquire()
            with h5py.File(hdf5_path, "a", libver="latest", swmr=True) as hdf5_file:
                if file_name not in hdf5_file:
                    dataset = hdf5_file.create_dataset(file_name, data=features.numpy().astype(np.float32))
                    # Store genres as a comma-separated string in the dataset
                    genre_string = ",".join(genres)
                    dataset.attrs["genre"] = genre_string
                    hdf5_file.flush()
        finally:
            lock.release()


        return file_name

    except OSError as e:
        print(f"❌ ERROR: Could not process {file_name}: {e}")
        return None


def extract_and_save_features(audio_dir, output_path):
    """Extract features using multiprocessing for speed."""
    print(f"\n🎵 Extracting features for: {audio_dir}")

    if not os.path.exists(audio_dir):
        print(f"❌ ERROR: Directory not found: {audio_dir}")
        return
  
    audio_files = get_all_audio_files(audio_dir)

    if not audio_files:
            print(f"⚠️ No MP3/WAV files found in {audio_dir}, skipping extraction.")
            return
    # Load already extracted file names from the HDF5 file
    extracted_files = set()
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as hdf5_file:
            extracted_files = set(hdf5_file.keys())

    # Filter out files that already have features extracted
    audio_files = [f for f in audio_files if os.path.basename(f) not in extracted_files]

    

    num_workers = max(8, multiprocessing.cpu_count() - 1)
    start_time = time()

    with Pool(processes=num_workers) as pool, tqdm(total=len(audio_files), unit="file") as pbar:
        for file_name in pool.imap_unordered(process_audio_wrapper, [(file, output_path) for file in audio_files ]):
            elapsed = time() - start_time
            remaining = (len(audio_files) - pbar.n) * (elapsed / (pbar.n + 1e-6))
            if pbar.n > 1:
                remaining = (len(audio_files) - pbar.n) * (elapsed / (pbar.n))
                rate = pbar.n / elapsed
            else:
                remaining = 0
                rate = 0.0
            pbar.set_description(f"🔄 {file_name} | {rate:.2f} files/sec | ETA: {remaining:.1f} sec")
            pbar.update(1)

    

if __name__ == "__main__":
    print("\n🔍 **Starting Feature Extraction Process...**")

    extract_and_save_features(USER_AUDIO_DIR, USER_FEATURES_PATH)
    extract_and_save_features(FMA_AUDIO_DIR, FMA_FEATURES_PATH)

    print("\n🎵✅ **Feature Extraction Process Complete!**")