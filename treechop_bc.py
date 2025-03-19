import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MineRLDataset(Dataset):
    def __init__(self, data_dir, save_path="processed_data.pt", rebuild=False):
        self.data = []
        if not rebuild and os.path.exists(save_path):
            print(f"Loading preprocessed dataset from {save_path}...")
            self.data = torch.load(save_path)
            print(f"Loaded {len(self.data)} frame-action pairs!")
        else:
            print(f"Processing dataset from raw videos...")
            # Get a list of all folders in dataset
            all_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
            
            print(f"Found {len(all_folders)} gameplay recordings. Loading data...")

            for folder in tqdm(all_folders, desc="Loading dataset"):
                folder_path = os.path.join(data_dir, folder)
                npz_file = os.path.join(folder_path, "rendered.npz")
                video_file = os.path.join(folder_path, "recording.mp4")

                if os.path.exists(npz_file) and os.path.exists(video_file):
                    data = np.load(npz_file)

                    # Load action labels
                    actions = {
                        "forward": data["action$forward"],
                        "left": data["action$left"],
                        "back": data["action$back"],
                        "right": data["action$right"],
                        "jump": data["action$jump"],
                        "sneak": data["action$sneak"],
                        "sprint": data["action$sprint"],
                        "attack": data["action$attack"],
                        "camera": data["action$camera"],  # Camera movement (Nx2)
                    }

                    # Extract frames from video with progress bar
                    frames = self.extract_frames(video_file, len(actions["forward"]))

                    # Ensure frames and actions are correctly aligned
                    if len(frames) == len(actions["forward"]):
                        for i in range(len(frames)):
                            self.data.append((frames[i], {key: actions[key][i] for key in actions}))
                    else:
                        print(f"Frame-action mismatch in {folder}: Frames={len(frames)}, Actions={len(actions['forward'])}")
        
        torch.save(self.data, save_path)
        print(f"Processed dataset saved at {save_path}! Total samples: {len(self.data)}")

    def extract_frames(self, video_path, expected_frames):
        """ Extracts frames from an MP4 video and resizes them to 64x64. """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (64, 64))  # Ensure consistent resolution
            frames.append(frame)

        cap.release()

        # Adjust frame count if necessary
        if len(frames) > expected_frames:
            frames = frames[:expected_frames]
        elif len(frames) < expected_frames:
            print(f"Warning: Frames ({len(frames)}) < Expected ({expected_frames}) in {video_path}")

        return frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame, actions = self.data[idx]

        # Convert frame to tensor and normalize
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (C, H, W)

        # Convert actions to tensor
        action_vector = torch.tensor([
            actions["forward"], actions["left"], actions["back"], actions["right"],
            actions["jump"], actions["sneak"], actions["sprint"], actions["attack"]
        ], dtype=torch.float32)

        # Camera movement as continuous values
        camera_movement = torch.tensor(actions["camera"], dtype=torch.float32)

        return frame, action_vector, camera_movement

class BCModel(nn.Module):
    def __init__(self):
        super(BCModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU()
        )
        
        # Predict movement actions (binary classification for each action)
        self.action_head = nn.Linear(256, 8)  # 8 movement actions
        
        # Predict camera movement (continuous values)
        self.camera_head = nn.Linear(256, 2)  # Camera X and Y

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        action_logits = self.action_head(x)  # Binary movement prediction
        camera_movement = self.camera_head(x)  # Continuous camera movement
        
        return action_logits, camera_movement
        

if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define dataset and dataloader
    data_dir = "data/MineRLTreechop-v0"
    save_path = "data/MineRLTreechop-v0/processed_data.pt"

    dataset = MineRLDataset(data_dir, save_path, rebuild=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

    def save_model(model, epoch, save_dir="models"):
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        save_path = os.path.join(save_dir, f"bc_model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

    # Initialize model, optimizer, and loss functions
    model = BCModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    action_loss_fn = nn.BCEWithLogitsLoss()  # For movement actions (binary)
    camera_loss_fn = nn.MSELoss()  # For continuous camera movement

    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Training Epoch {epoch+1}/{num_epochs}...")
        
        for frames, actions, cameras in tqdm(dataloader, desc=f"Training {epoch+1}/{num_epochs}"):
            # Move batch data to GPU
            frames, actions, cameras = frames.to(device), actions.to(device), cameras.to(device)

            optimizer.zero_grad()

            # Forward pass
            action_logits, camera_preds = model(frames)

            # Compute loss
            action_loss = action_loss_fn(action_logits, actions)
            camera_loss = camera_loss_fn(camera_preds, cameras)
            loss = action_loss + camera_loss  # Total loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} completed. Loss: {total_loss:.4f}\n")
        save_model(model, epoch+1, "models/treechop_bc_test_1")
