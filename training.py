import torch
import gc
import traceback
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import device, batch_size, num_workers, DEBUG
from dataset import compute_class_weights, load_genre_index
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from dataset import compute_class_weights
from config import USER_FEATURES_PATH, FMA_FEATURES_PATH, USER_INDEX, COMBINED_INDEX, gradient_accumulation_steps
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset
import random
from torch.utils.data import Subset
import random

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

GENRE_TO_INDEX, INDEX_TO_GENRE = load_genre_index(COMBINED_INDEX)

def soft_genre_mask_from_index(user_index, total_genres):
    mask = torch.ones(total_genres, dtype=torch.float32)
    user_genre_ids = set(user_index.values())

    for i in range(total_genres):
        if i not in user_genre_ids:
            mask[i] = 0.1  # or whatever low weight you want
    return mask.to(device)



# In training.py, fix enhanced_metric_diagnostics function
def enhanced_metric_diagnostics(all_preds, all_targets, thresholds=[0.3,0.5,0.7]):
    """
    Comprehensive metric diagnostics for multi-label classification
    """
    # Concatenate predictions and targets
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Apply sigmoid to raw predictions
    sigmoid_preds = torch.sigmoid(preds)
    
    if DEBUG:
        print("\n🔍 Multi-Label Classification Diagnostics 🔍")
        print("\n📊 Prediction and Target Distribution:")
        print(f"Total Samples: {len(targets)}")
        print(f"Total Genres: {targets.shape[1]}")
        
    if DEBUG:
        target_counts = targets.sum(dim=0)
        print("\n🏷️ Genre Label Counts:")
        for i, count in enumerate(target_counts):
            genre = INDEX_TO_GENRE.get(i, f"Genre {i}")
            print(f"{genre}: {count.item()} samples")
        
    # Prediction Analysis per Threshold
    print("\n🎯 Performance Across Thresholds:")
    full_metrics = {}
    
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        binary_preds = (sigmoid_preds > threshold).float()
        
        # Per-Class Metrics
        per_class_metrics = {}
        for i in range(targets.shape[1]):
            genre = INDEX_TO_GENRE.get(i, f"Genre {i}")
            tp = ((binary_preds[:, i] == 1) & (targets[:, i] == 1)).sum().item()
            fp = ((binary_preds[:, i] == 1) & (targets[:, i] == 0)).sum().item()
            fn = ((binary_preds[:, i] == 0) & (targets[:, i] == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[genre] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        if DEBUG:
            print("\nPer-Genre Performance:")
            for genre, metrics in per_class_metrics.items():
                print(f"{genre}: {metrics}")
            
        # Micro and Macro Averages
        macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in per_class_metrics.values()])
        from sklearn.metrics import precision_recall_fscore_support

        binary_preds_np = binary_preds.numpy()
        targets_np = targets.numpy()

        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            targets_np, binary_preds_np, average='micro', zero_division=0
        )

        print(f"Micro F1: {micro_f1:.4f}")

            
        if DEBUG:
            print(f"\nMacro Averages (Threshold {threshold}):")
            print(f"Macro Precision: {macro_precision:.4f}")
            print(f"Macro Recall: {macro_recall:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")

        full_metrics[threshold] = {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class': per_class_metrics
        }
        
    return full_metrics  # Return after processing all thresholds

# Modify your training script to call this function
# Replace existing metric calculation with this diagnostic

def train_model(model_manager, user_dataset, fma_dataset, user_index, combined_index, epochs_stage_1=5, epochs_stage_2=5):
    """Train in two stages: (1) personal dataset, (2) combined dataset."""
  

    model_manager.model.train()

    class_weights = compute_class_weights([USER_FEATURES_PATH, FMA_FEATURES_PATH], GENRE_TO_INDEX)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Update criterion with class weights
    print("🔧 Updating Loss Criterion...")
    model_manager.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
   
    best_f1_score = 0.0
    patience = 10
    epochs_no_improve = 0

    soft_mask = soft_genre_mask_from_index(user_index, len(combined_index))

    


    for stage, stage_datasets, epochs in [
        (1, user_dataset, epochs_stage_1),
        (2, torch.utils.data.ConcatDataset([user_dataset, fma_dataset]), epochs_stage_2)
    ]:
        print(f"\n🎵 **Stage {stage}: Training 🎵")
        print(f"Stage {stage} Dataset Size: {len(stage_datasets)}")
        if stage == 2:
            print("🔁 Switching to combined dataset...")
            # (Optional) Rebuild class weights
            class_weights = compute_class_weights([USER_FEATURES_PATH, FMA_FEATURES_PATH], GENRE_TO_INDEX)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            model_manager.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

            # Recompute soft mask
            soft_mask = soft_genre_mask_from_index(user_index, len(combined_index))

        dataloader = DataLoader(
            stage_datasets,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        for local_epoch in range(epochs):
            epoch = model_manager.current_epoch + local_epoch

            print(f"Starting Epoch {epoch+1}")
            
            model_manager.model.train()
            total_loss = 0
            all_preds = []
            all_targets = []

            # Add a check to ensure dataloader is not empty
            if len(dataloader) == 0:
                print(f"⚠️ Warning: Empty dataloader in Stage {stage}, Epoch {epoch+1}")
                continue

            with tqdm(dataloader, desc=f"🚀 Stage {stage} - Epoch {epoch+1}/{epochs}") as pbar:
                model_manager.optimizer.zero_grad() 
                for batch_idx, (features, labels) in enumerate(pbar):
                    try:
                        # Ensure contiguous and correct shape
                        features = features.float().to(device)
                        labels = labels.float().to(device)
                        
                    
                        outputs = model_manager.model(features)
                     
                        raw_loss = model_manager.criterion(outputs, labels)  # [B, C]
                        weighted_loss = raw_loss * class_weights.unsqueeze(0)  # Apply class weights
                        masked_loss = weighted_loss * soft_mask.unsqueeze(0)  # Apply soft masking
                        loss = masked_loss.mean()  # or use .sum() / batch_size

                        if DEBUG:
                            sig_out = torch.sigmoid(outputs)
                            top_pred = torch.topk(sig_out, 3, dim=1).indices  # top 3 predicted genres per sample
                            top_probs = torch.topk(sig_out, 3, dim=1).values

                            
                            print(f"🧠 Predicted Genres for Sample 0:")
                            for i, (genre_idx, prob) in enumerate(zip(top_pred[0], top_probs[0])):
                                genre_name = INDEX_TO_GENRE.get(genre_idx.item(), f"Genre {genre_idx.item()}")
                                print(f"  {i+1}. {genre_name} ({prob.item():.2f})")

                            # 🏷️ DEBUG: Print actual ground truth genres for Sample 0
                            true_indices = (labels[0] == 1.0).nonzero(as_tuple=True)[0]
                            true_genres = [INDEX_TO_GENRE[i.item()] for i in true_indices]

                            print(f"🏷️ True Genres for Sample 0:")
                            for genre in true_genres:
                                print(f"  ✔ {genre}")



                        # Use the scaler for mixed precision training
                        model_manager.scaler.scale(loss).backward()
                        # Only zero gradients at the start of accumulation steps:
                        if batch_idx % gradient_accumulation_steps == 0:
                            model_manager.optimizer.zero_grad()
                            
                        # After backward:
                        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                            model_manager.scaler.step(model_manager.optimizer)
                            model_manager.scaler.update()
                                            

                    

                        # Collect predictions and targets for metrics
                        all_preds.append(torch.sigmoid(outputs).detach().cpu())
                        all_targets.append(labels.detach().cpu())

                        total_loss += loss.item()
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                        
                        # Memory management
                        del features, labels, outputs, loss
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {e}")
                        traceback.print_exc()
                        continue

            # Check if we have any predictions to process
            if not all_preds or not all_targets:
                print(f"⚠️ No predictions collected in Stage {stage}, Epoch {epoch+1}")
                continue


            try:
               
                metrics = enhanced_metric_diagnostics(all_preds, all_targets)
                best_threshold, best_threshold_metrics = max(
                    metrics.items(), key=lambda x: x[1]['macro_f1']
                )
                
                # ✅ Proceed to calculate/save as before...
                f1_value = best_threshold_metrics['macro_f1']
                
                print(f"   Best Threshold: {best_threshold}")
                print(f"✅ Stage {stage} - Epoch {epoch+1}")
                print(f"   Loss: {total_loss / len(dataloader):.4f}")
                print(f"   Best Threshold Metrics:")
                for metric, value in best_threshold_metrics.items():
                    if isinstance(value, dict):
                        continue 
                    print(f"   {metric.capitalize()}: {value:.4f}")

                sigmoid_preds = torch.sigmoid(torch.cat(all_preds, dim=0))
                mean_scores = sigmoid_preds.mean(dim=0)
                top_indices = mean_scores.topk(5).indices
                print("\n📈 Most Predicted Genres This Epoch:")
                for i in top_indices:
                    genre = INDEX_TO_GENRE.get(i.item(), f"Genre {i.item()}")
                    print(f"  {genre} → avg score: {mean_scores[i]:.3f}")


                if f1_value > best_f1_score:
                    print(f"📈 New best F1: {f1_value:.4f} (previous: {best_f1_score:.4f})")
                    best_f1_score = f1_value
                    epochs_no_improve = 0
                    model_manager.save_model()
                else:
                    print(f"⏸ No F1 improvement: {f1_value:.4f} vs best {best_f1_score:.4f}")
                    epochs_no_improve += 1


                if epochs_no_improve >= patience:
                    print(f"\n⏹️ Early stopping: No improvement for {patience} epochs")
                    break


                model_manager.scheduler.step(f1_value)
                model_manager.save_checkpoint()

            except Exception as e:
                print(f"Error processing metrics in Stage {stage}, Epoch {epoch+1}: {e}")
                traceback.print_exc()

    print("\n✅ Training Complete!")