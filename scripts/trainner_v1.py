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

# Dataset Class
class PairedAudioDataset(Dataset):
    def __init__(self, folder_path, sample_rate=96000, segment_length=1.0):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder {folder_path} not found.")
        self.folder_path = folder_path
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(sample_rate * segment_length)

        # Find paired files
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

        # Load low-quality and high-quality audio
        low_audio, _ = librosa.load(low_path, sr=self.sample_rate, mono=True)
        high_audio, _ = librosa.load(high_path, sr=self.sample_rate, mono=True)

        # Ensure both files are the same length
        min_length = min(len(low_audio), len(high_audio))
        low_audio = low_audio[:min_length]
        high_audio = high_audio[:min_length]

        # Extract a random segment
        if len(low_audio) > self.segment_samples:
            start = np.random.randint(0, len(low_audio) - self.segment_samples)
            end = start + self.segment_samples
            low_audio = low_audio[start:end]
            high_audio = high_audio[start:end]

        # Convert to PyTorch tensors
        low_tensor = torch.tensor(low_audio, dtype=torch.float32)
        high_tensor = torch.tensor(high_audio, dtype=torch.float32)

        return low_tensor, high_tensor

# Model Class
class AudioEnhancementModel(nn.Module):
    def __init__(self):
        super(AudioEnhancementModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=9, stride=1, padding=4),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)  # Remove channel dimension

# Initialize weights
def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Training Function
def train_model(model, dataloader, val_dataloader, epochs, device, learning_rate=1e-8, stp_size=10, sch_gamma=0.5, opt_w_dcy=1e-10, eps_rate=1e-8):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=opt_w_dcy, eps=eps_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=stp_size, gamma=sch_gamma)
    model.apply(init_weights)  # Apply weight initialization
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for low_quality, high_quality in dataloader:
            low_quality, high_quality = low_quality.to(device), high_quality.to(device)
            optimizer.zero_grad()  # Zero gradients
            outputs = model(low_quality)
            loss = criterion(outputs, high_quality)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for low_quality, high_quality in val_dataloader:
                low_quality, high_quality = low_quality.to(device), high_quality.to(device)
                outputs = model(low_quality)
                loss = criterion(outputs, high_quality)
                val_loss += loss.item()

        loss_data.append({
            "tl": total_loss / len(dataloader),
            "vl": val_loss / len(val_dataloader),
            "it": epoch
        })
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss_data[-1]['tl']:.6f}, Validation Loss: {loss_data[-1]['vl']:.6f}")

# Model Control Function
def model_ctrl(b_size=64, t_frac=0.8, lr=1e-8, sp_size=5, m_gamma=0.5, epoch_steps=32, samp_rate=44100, dk_rate=1e-25, eps_rate=1e-8):
    # Data Loader
    folder_path = r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\trainning_data"
    dataset = PairedAudioDataset(folder_path)
    train_size = int(t_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size, shuffle=False)

    # Initialize and Train the Model
    model = AudioEnhancementModel()

    train_model(
        model,
        train_dataloader,
        val_dataloader,
        epochs=epoch_steps,
        device=device,
        learning_rate=lr,
        stp_size=sp_size,
        sch_gamma=m_gamma,
        opt_w_dcy=dk_rate,
        eps_rate=eps_rate
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
            'segment_length': 1.0
        },
    }, model_path)
    print("Model and hyperparameters saved.")

# Execute the Control Function
model_ctrl(
    b_size=1024,
    t_frac=1.0,
    lr=1e-5,
    sp_size=20,
    m_gamma=0.8,
    epoch_steps=64,
    samp_rate=sr,
    dk_rate=0,
    eps_rate=1e-6
)
