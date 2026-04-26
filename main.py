import multiprocessing as mp
import traceback
from dataset import load_datasets
from training import train_model
from model import ModelManager
from classifier import classify_and_sort
from torch.utils.data import Subset
import random
from config import USER_FEATURES_PATH, FMA_FEATURES_PATH

if __name__ == "__main__":
    try:
       
        mp.set_start_method("spawn", force=True)
        
        user_dataset, fma_dataset, user_index, combined_index = load_datasets()


            # user_dataset = Subset(user_dataset, random.sample(range(len(user_dataset)), 30))             
        if user_dataset is None or fma_dataset is None:
            print("❌ ERROR: One or both datasets failed to load. Stopping execution.")
            exit(1)
        
        model_manager = ModelManager(num_genres=len(combined_index))

        # ✅ Try to load checkpoint
        if model_manager.load_checkpoint():
            print(f"✅ Resuming training from epoch {model_manager.current_epoch}")
        else:
            print("🆕 Starting training from scratch")

        # ✅ Train the model
        train_model(model_manager, user_dataset, fma_dataset, user_index, combined_index)

        print("\n✅ Training complete! Now classifying and sorting new songs...")
        classify_and_sort(model_manager)  # This now uses the function from classifier.py
        print("\n🎵✅ All songs have been classified and sorted!")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        traceback.print_exc()