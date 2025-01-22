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
    def __init__(self, folder_path, sample_rate=96000, segment_length=0.25):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder {folder_path} not found.")
        self.folder_path = folder_path
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(sample_rate * segment_length)

        self.pairs = []
        for file in os.listdir(folder_path):
            if file.endswith("_low.wav"):
                pair_id = file.replace("_low.wav", "")
                high_file = f"{pair_id}_high.wav"
                if high_file in os.listdir(folder_path):
                    self.pairs.append((file, high_file))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_file, high_file = self.pairs[idx]
        low_path = os.path.join(self.folder_path, low_file)
        high_path = os.path.join(self.folder_path, high_file)

        low_audio, _ = librosa.load(low_path, sr=self.sample_rate, mono=True)
        high_audio, _ = librosa.load(high_path, sr=self.sample_rate, mono=True)

        low_audio = low_audio.astype(np.float32) / (np.max(np.abs(low_audio)) + 1e-8)
        high_audio = high_audio.astype(np.float32) / (np.max(np.abs(high_audio)) + 1e-8)

        min_length = min(len(low_audio), len(high_audio))
        low_audio = low_audio[:min_length]
        high_audio = high_audio[:min_length]

        if len(low_audio) > self.segment_samples:
            start = np.random.randint(0, len(low_audio) - self.segment_samples)
            end = start + self.segment_samples
            low_audio = low_audio[start:end]
            high_audio = high_audio[start:end]

        low_tensor = torch.tensor(low_audio, dtype=torch.float32)
        high_tensor = torch.tensor(high_audio, dtype=torch.float32)

        return low_tensor, high_tensor


def collate_fn(batch):
    return [b for b in batch if b is not None]


class AudioEnhancementModel(nn.Module):
    def __init__(self):
        super(AudioEnhancementModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=9, stride=1, padding=4),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)


def train_model(model, dataloader, val_dataloader, epochs, device, learning_rate=1e-5):
    criterion = nn.MSELoss()
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
                loss = criterion(outputs, high_quality)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

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

        loss_data.append({"tl": total_loss / len(dataloader), "vl": val_loss / len(val_dataloader)})
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss_data[-1]['tl']:.6f}, Validation Loss: {loss_data[-1]['vl']:.6f}")


# Model Control Function
def model_ctrl(b_size=32, t_frac=0.8, lr=1e-5, epoch_steps=32, samp_rate=44100):
    # Data Loader
    folder_path = r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\trainning_data"
    print(f"Checking dataset folder: {folder_path}")
    if not os.path.exists(folder_path):
        print("Dataset folder does not exist!")
        return

    dataset = PairedAudioDataset(folder_path)
    print(f"Number of pairs in dataset: {len(dataset)}")

    train_size = int(t_frac * len(dataset))
    val_size = len(dataset) - train_size
    print(f"Training set size: {train_size}, Validation set size: {val_size}")

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    optimal_batch_size = min(b_size, len(train_dataset), len(val_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=optimal_batch_size,
        shuffle=True,
        num_workers=0,  # Single-threaded loading
        pin_memory=False,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=optimal_batch_size,
        shuffle=False,
        num_workers=0,  # Single-threaded loading
        pin_memory=False,
        collate_fn=collate_fn
    )

    # Initialize and Train the Model
    model = AudioEnhancementModel()

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
            'segment_length': 0.25  # Adjusted segment length
        },
    }, model_path)
    print("Model and hyperparameters saved.")

# Execute the Control Function
if __name__ == "__main__":
    model_ctrl(
        b_size=256,
        t_frac=0.9,
        lr=1e-4,
        epoch_steps=32,
        samp_rate=sr
    )
