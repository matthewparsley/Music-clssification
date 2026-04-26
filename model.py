import os
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
from config import BASE_DIR, device


import torch.utils.checkpoint as checkpoint  

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# In model.py, modify GenreCNNLSTM class
class GenreCNNLSTM(nn.Module):
    def __init__(self, num_genres):
        super(GenreCNNLSTM, self).__init__()

        # CNN Feature Extractor with Residual Connections
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Create projection layer during initialization, not during forward pass
        self.norm = nn.LayerNorm(4096)  # Fixed size based on your 64x64 feature shape
        self.projection = nn.Linear(4096, 64)

        # LSTM for temporal sequence learning
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        # Fully Connected Layers for Classification
        self.fc_layers = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(128),
            nn.Linear(128, num_genres)
        )

    def forward(self, x):
        # [B, 64, T] → [B, 1, 64, T]
        x = x.view(x.shape[0], 1, 64, -1).contiguous()

        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))    # [B, 32, 64, T]
        x = F.relu(self.bn2(self.conv2(x)))    # [B, 64, 64, T]

        # Rearrange for LSTM: collapse H x W
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, 64, 64]
        x = x.view(x.shape[0], x.shape[1], -1)  # [B, T, 4096] ← flatten
        x = self.norm(x)
        x = self.projection(x)                  # [B, T, 64]

        # LSTM
        x, _ = self.lstm(x)                     # [B, T, 256]
        x = x[:, -1, :]                         # [B, 256] ← use last time step

        x = self.fc_layers(x)                   # [B, num_genres]
        return x



class ModelManager:
    def train_model(self, user_dataloader, fma_dataloader):
        from training import train_model
        train_model(self, user_dataloader, fma_dataloader) 
    def __init__(self, num_genres):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GenreCNNLSTM(num_genres)  # ✅ Uses new CNN+LSTM model
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model_path = os.path.join(BASE_DIR, "genre_classifier.pth")
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.002,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.scaler = torch.amp.GradScaler()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

        self.current_epoch = 0
        self.best_loss = float("inf") # Changed from float("inf") to 0.0 since higher F1 is better

    def save_model(self):
        """Save the trained model"""
        try:
            torch.save(self.model.state_dict(), self.model_path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
            traceback.print_exc()

    def load_model(self):
        """Load a trained model from file"""
        try:
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                print("Model loaded successfully")
            else:
                print("No saved model found, initializing new model")
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()

    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }
        checkpoint_path = os.path.join(BASE_DIR, "checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self):
        """Load training checkpoint"""
        checkpoint_path = os.path.join(BASE_DIR, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_loss = checkpoint['best_loss']
            return True
        return False

