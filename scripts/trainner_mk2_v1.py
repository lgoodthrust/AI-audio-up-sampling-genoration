import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sr = 64000
loss_data = []


class PairedAudioDataset(Dataset):
    def __init__(self, folder_path, sample_rate=96000, segment_length=0.1, mode="train"):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder {folder_path} not found.")
        self.folder_path = folder_path
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(sample_rate * segment_length)
        self.mode = mode

        self.pairs = []
        for file in os.listdir(folder_path):
            if file.endswith("_low.wav") and self.mode == "train":
                pair_id = file.replace("_low.wav", "")
                high_file = f"{pair_id}_high.wav"
                if high_file in os.listdir(folder_path):
                    self.pairs.append((file, high_file))
            elif file.endswith("_high.wav") and self.mode == "val":
                pair_id = file.replace("_high.wav", "")
                low_file = f"{pair_id}_low.wav"
                if low_file in os.listdir(folder_path):
                    self.pairs.append((low_file, file))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            low_file, high_file = self.pairs[idx]
            low_path = os.path.join(self.folder_path, low_file)
            high_path = os.path.join(self.folder_path, high_file)

            low_audio, _ = librosa.load(low_path, sr=self.sample_rate, mono=True)
            high_audio, _ = librosa.load(high_path, sr=self.sample_rate, mono=True)

            # Log audio statistics
            print(f"Loaded low_audio max: {np.max(np.abs(low_audio))}, high_audio max: {np.max(np.abs(high_audio))}")

            if np.max(np.abs(low_audio)) < 1e-8 or np.max(np.abs(high_audio)) < 1e-8:
                print(f"Skipping pair due to low max value: {low_file}, {high_file}")
                return None

            low_audio = low_audio.astype(np.float32) / (np.max(np.abs(low_audio)) + 1e-6)
            high_audio = high_audio.astype(np.float32) / (np.max(np.abs(high_audio)) + 1e-6)

            min_length = min(len(low_audio), len(high_audio))
            low_audio = low_audio[:min_length]
            high_audio = high_audio[:min_length]

            if len(low_audio) > self.segment_samples:
                start = np.random.randint(0, len(low_audio) - self.segment_samples)
                end = start + self.segment_samples
                low_audio = low_audio[start:end]
                high_audio = high_audio[start:end]

            low_tensor = torch.clamp(torch.tensor(low_audio, dtype=torch.float32), min=-1.0, max=1.0)
            high_tensor = torch.clamp(torch.tensor(high_audio, dtype=torch.float32), min=-1.0, max=1.0)

            return low_tensor, high_tensor
        except Exception as e:
            print(f"Error processing pair {idx}: {e}")
            return None


def collate_fn(batch):
    return [b for b in batch if b is not None]


class AudioEnhancementModel(nn.Module):
    def __init__(self):
        super(AudioEnhancementModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.LayerNorm([6400]),  # Match the last dimension of the input tensor
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.LayerNorm([3200]),  # Adjust dynamically for the next layer's output shape
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.LayerNorm([6400]),  # Match the last dimension of the decoder's input
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=9, stride=1, padding=4),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        print(f"Encoder output min: {x.min().item()}, max: {x.max().item()}")
        x = self.decoder(x)
        return x.squeeze(1)


def initialize_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_model(model, dataloader, val_dataloader, epochs, device, learning_rate=1e-6):
    criterion = nn.L1Loss()  # Changed to L1 loss for stability
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            low_quality, high_quality = zip(*batch)
            low_quality = torch.stack(low_quality).to(device)
            high_quality = torch.stack(high_quality).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(low_quality)
                loss = criterion(outputs + 1e-8, high_quality)

            # Log outputs for debugging
            print(f"Outputs min: {outputs.min().item()}, max: {outputs.max().item()}, mean: {outputs.mean().item()}")

            # Skip update if loss is NaN
            if torch.isnan(loss):
                print("NaN detected in loss. Skipping update.")
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced gradient clipping value
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # Check for NaN in model weights
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in {name}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                low_quality, high_quality = zip(*batch)
                low_quality = torch.stack(low_quality).to(device)
                high_quality = torch.stack(high_quality).to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(low_quality)
                    val_loss += criterion(outputs, high_quality).item()

                # Log validation loss per batch
                print(f"Validation batch loss: {val_loss}")

        loss_data.append({"tl": total_loss / len(dataloader), "vl": val_loss / len(val_dataloader)})
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss_data[-1]['tl']:.6f}, Validation Loss: {loss_data[-1]['vl']:.6f}")


# Model Control Function
def model_ctrl(b_size=32, lr=1e-6, epoch_steps=32, samp_rate=44100):
    # Data Loader
    folder_path = r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\trainning_data"
    print(f"Checking dataset folder: {folder_path}")
    if not os.path.exists(folder_path):
        print("Dataset folder does not exist!")
        return

    train_dataset = PairedAudioDataset(folder_path, sample_rate=samp_rate, mode="train")
    val_dataset = PairedAudioDataset(folder_path, sample_rate=samp_rate, mode="val")

    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=min(b_size, len(train_dataset)),  # Adjust batch size dynamically
        shuffle=True,
        num_workers=0,  # Single-threaded loading
        pin_memory=False,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=min(b_size, len(val_dataset)),  # Adjust batch size dynamically
        shuffle=False,
        num_workers=0,  # Single-threaded loading
        pin_memory=False,
        collate_fn=collate_fn
    )

    # Initialize and Train the Model
    model = AudioEnhancementModel()
    model.apply(initialize_weights)

    train_model(
        model,
        train_dataloader,
        val_dataloader,
        epochs=epoch_steps,
        device=device,
        learning_rate=lr,
    )

    # Initialize lists to store training and validation losses
    training_loss = [entry["tl"] for entry in loss_data]
    validation_loss = [entry["vl"] for entry in loss_data]
    epochs = range(len(loss_data))

    plt.plot(epochs, training_loss, label="Training Loss")
    plt.plot(epochs, validation_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    # Save Model and Hyperparameters
    model_path = r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\models\01.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'batch_size': b_size,
            'learning_rate': lr,
            'sample_rate': samp_rate,
            'segment_length': 0.1  # Adjusted segment length
        },
    }, model_path)
    print("Model and hyperparameters saved.")

# Execute the Control Function
if __name__ == "__main__":
    model_ctrl(
        b_size=256,
        lr=1e-6,
        epoch_steps=32,
        samp_rate=sr
    )
