import os
import torch
import warnings
import logging
import torchaudio

DEBUG = True  # Set to True if you want to see debug messages

logging.getLogger('torchaudio').setLevel(logging.ERROR)
logging.getLogger('librosa').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning) 
os.environ['PYTHONWARNINGS'] = 'ignore'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torchaudio.set_audio_backend("soundfile") 

torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_float32_matmul_precision('medium')  

device = torch.device("cuda:0")  # ✅ Force PyTorch to use GPU 0 (NVIDIA)

batch_size = 3 # ✅ Increase batch size for faster training
num_workers = 4
gradient_accumulation_steps = 2  # 3*2 = effective batch size of 6


BASE_DIR = "/home/mattp69/Desktop/Music_sorter"
USER_AUDIO_DIR = os.path.join(BASE_DIR, "Training/Personal")
FMA_AUDIO_DIR = os.path.join(BASE_DIR, "Training/Fma_medium/fma_medium")
UNSORTED_DIR = os.path.join(BASE_DIR, "Unsorted")
SORTED_DIR = os.path.join(BASE_DIR, "Sorted")
USER_FEATURES_PATH = os.path.join(BASE_DIR, "Training/Features/user_features.h5")
FMA_FEATURES_PATH = os.path.join(BASE_DIR, "Training/Features/fma_features.h5")
USER_GENRE_FILE = os.path.join(BASE_DIR, "Code/Dataset/user_genre_mapping.json")
FMA_GENRE_FILE = os.path.join(BASE_DIR, "Code/Dataset/fma_genre_mapping.json")
USER_INDEX = os.path.join(BASE_DIR, "Code/Dataset/user_genre_index.json")
COMBINED_INDEX = os.path.join(BASE_DIR, "Code/Dataset/combined_genre_index.json")


SAMPLE_RATE = 16000