import os
import torch
import shutil
import gc
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from config import device, UNSORTED_DIR, SORTED_DIR
from config import COMBINED_INDEX
from dataset import load_genre_index
from feature_extractor import extract_features

GENRE_TO_INDEX, INDEX_TO_GENRE = load_genre_index(COMBINED_INDEX)
class MusicClassifier:
    def __init__(self, model_manager):
        self.model = model_manager.model
        
    def process_file(self, file):
        """Process a single music file for classification"""
        if not file.endswith(".mp3"):
            return

        file_path = os.path.join(UNSORTED_DIR, file)
        features = extract_features(file_path)

        if features is None:
            print(f"Could not extract features from {file}")
            return

        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = self.model(features_tensor)
            
            # Apply sigmoid for multi-label inference
            outputs = torch.sigmoid(outputs)
            
            # Get top genres with probability > 0.5
            predicted_labels = (outputs > 0.5).squeeze().nonzero(as_tuple=True)[0]

            if len(predicted_labels) == 0:
                # If no genres have probability > 0.5, just take the top genre
                _, predicted_labels = torch.topk(outputs.squeeze(), 1)

            genre_folders = [INDEX_TO_GENRE[label.item()] for label in predicted_labels]

            for genre in genre_folders:
                genre_folder_path = os.path.join(SORTED_DIR, genre)
                os.makedirs(genre_folder_path, exist_ok=True)
                shutil.copy(file_path, os.path.join(genre_folder_path, file))

            print(f"✅ Moved {file} to folders: {', '.join(genre_folders)}")

        torch.cuda.empty_cache()
        gc.collect()
        
    def classify_and_sort(self):
        """Use the trained model to classify and sort music into folders."""
        if not os.path.exists(UNSORTED_DIR):
            print("Unsorted folder missing!")
            return

        os.makedirs(SORTED_DIR, exist_ok=True)
        self.model.eval()

        files = [f for f in os.listdir(UNSORTED_DIR) if f.endswith(".mp3")]
        
        if not files:
            print("No MP3 files found in the unsorted directory!")
            return

        # ✅ Process files in small batches (e.g., 10 at a time)
        batch_size = 2  
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]

            with ThreadPoolExecutor(max_workers=3) as executor:
                list(tqdm(executor.map(self.process_file, batch), 
                          desc="Sorting files", 
                          total=len(batch)))

            torch.cuda.empty_cache()  # ✅ Free CUDA memory
            gc.collect()  # ✅ Free CPU memory

# Helper function to use in main.py
def classify_and_sort(model_manager):
    classifier = MusicClassifier(model_manager)
    classifier.classify_a