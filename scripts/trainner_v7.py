import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

# If needed, install external libraries for evaluation metrics:
# pip install pystoi

sr = 96000
loss_data = []

try:
    from pystoi import stoi
except ImportError:
    stoi = None
    print("pystoi not installed. STOI metric will be skipped.")

try:
    from pesq import pesq
except ImportError:
    pesq = None
    print("pesq not installed. PESQ metric will be skipped.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################
# Multi-Resolution STFT Loss (optional)
########################################
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048], hop_sizes=[256, 512], win_lengths=[1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, enhanced, target):
        loss_val = 0.0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            enhanced_stft = torch.stft(
                enhanced, n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                return_complex=True,
                window=torch.hann_window(
                        window_length=512,
                        periodic=False,
                        dtype=torch.float32,
                        device=device,
                        pin_memory=False,
                        requires_grad=False
                )
            )

            target_stft = torch.stft(
                target, n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                return_complex=True,
                window=torch.hann_window(
                        window_length=512,
                        periodic=False,
                        dtype=torch.float32,
                        device=device,
                        pin_memory=False,
                        requires_grad=False
                )
            )
            # Simple spectral L1
            loss_val += (enhanced_stft - target_stft).abs().mean()
        # Average over the number of STFT scales
        return loss_val / len(self.fft_sizes)

#####################################################################
# 1. Model Architecture: U-Net style 1D network with skip connections
#    + Additional depth + a dilated convolution layer for broader receptive field.
#####################################################################
class AudioEnhancementModel(nn.Module):
    def __init__(self):
        super(AudioEnhancementModel, self).__init__()

        # --- Encoder ---
        self.enc_conv1 = nn.Conv1d(1, 128, kernel_size=15, stride=1, padding=7)
        self.enc_bn1   = nn.BatchNorm1d(128)

        self.enc_conv2 = nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7)
        self.enc_bn2   = nn.BatchNorm1d(256)

        self.enc_conv3 = nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7)
        self.enc_bn3   = nn.BatchNorm1d(512)

        self.enc_conv4 = nn.Conv1d(512, 1024, kernel_size=15, stride=2, padding=7)
        self.enc_bn4   = nn.BatchNorm1d(1024)

        # --- Bottleneck (Stacked Conv layers) ---
        self.bottleneck_conv1 = nn.Conv1d(1024, 1024, kernel_size=15, stride=1, padding=7)
        self.bottleneck_bn1   = nn.BatchNorm1d(1024)

        self.bottleneck_conv2 = nn.Conv1d(1024, 1024, kernel_size=15, stride=1, padding=7)
        self.bottleneck_bn2   = nn.BatchNorm1d(1024)

        # --- Decoder ---
        self.dec_deconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.dec_bn4     = nn.BatchNorm1d(512)

        self.dec_deconv3 = nn.ConvTranspose1d(512, 256, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.dec_bn3     = nn.BatchNorm1d(256)

        self.dec_deconv2 = nn.ConvTranspose1d(256, 128, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.dec_bn2     = nn.BatchNorm1d(128)

        self.dec_conv1   = nn.Conv1d(128, 1, kernel_size=15, stride=1, padding=7)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,T)

        # --- Encoder ---
        e1 = self.relu(self.enc_bn1(self.enc_conv1(x)))   # (B,128,T)
        e2 = self.relu(self.enc_bn2(self.enc_conv2(e1)))  # (B,256,T/2)
        e3 = self.relu(self.enc_bn3(self.enc_conv3(e2)))  # (B,512,T/4)
        e4 = self.relu(self.enc_bn4(self.enc_conv4(e3)))  # (B,1024,T/8)

        # --- Bottleneck ---
        b  = self.relu(self.bottleneck_bn1(self.bottleneck_conv1(e4)))  # (B,1024,T/8)
        b  = self.relu(self.bottleneck_bn2(self.bottleneck_conv2(b)))  # (B,1024,T/8)

        # --- Decoder (Skip Connections) ---
        d4 = self.relu(self.dec_bn4(self.dec_deconv4(b)))  # (B,512,T/4)
        d4 = d4 + e3[:, :, :d4.shape[2]]  # Skip connection

        d3 = self.relu(self.dec_bn3(self.dec_deconv3(d4)))  # (B,256,T/2)
        d3 = d3 + e2[:, :, :d3.shape[2]]  # Skip connection

        d2 = self.relu(self.dec_bn2(self.dec_deconv2(d3)))  # (B,128,T)
        d2 = d2 + e1[:, :, :d2.shape[2]]  # Skip connection

        out = self.tanh(self.dec_conv1(d2))  # Output in range [-1, 1]
        out = out.squeeze(1)  # (B,T)
        return out

#####################################################################
# 2. Data Preprocessing & Augmentation
#####################################################################
def random_pitch_shift(audio, sample_rate, max_steps=2.0):
    """
    Randomly shift the pitch of the audio by some semitones in [-max_steps, max_steps].
    """
    steps = np.random.uniform(-max_steps, max_steps)
    return librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=steps)

def random_time_stretch(audio, max_rate_deviation=0.15):
    """
    Randomly stretch/shrink time by a factor in [1-max_rate_deviation, 1+max_rate_deviation].
    """
    rate = np.random.uniform(1 - max_rate_deviation, 1 + max_rate_deviation)
    return librosa.effects.time_stretch(y=audio, rate=rate)

class PairedAudioDataset(Dataset):
    def __init__(
        self, folder_path, sample_rate=96000,
        segment_length=0.25, apply_augmentation=False
    ):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder {folder_path} not found.")
        self.folder_path = folder_path
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(sample_rate * segment_length)
        self.apply_augmentation = apply_augmentation

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

        # Normalize to [-1,1]
        low_audio = low_audio.astype(np.float32)
        low_audio = low_audio / (np.max(np.abs(low_audio)) + 1e-10)
        high_audio = high_audio.astype(np.float32)
        high_audio = high_audio / (np.max(np.abs(high_audio)) + 1e-10)


        # OPTIONAL data augmentation on low_audio only
        if self.apply_augmentation:
            if np.random.uniform(0.0,1.0) < 0.5:
                low_audio = random_pitch_shift(low_audio, self.sample_rate, max_steps=2.0)
            if np.random.uniform(0.0,1.0) < 0.5:
                try:
                    low_audio = random_time_stretch(low_audio, max_rate_deviation=0.15)
                except:
                    pass  # If the segment gets too short, skip

        # Make sure both are the same length
        min_length = min(len(low_audio), len(high_audio))
        low_audio = low_audio[:min_length]
        high_audio = high_audio[:min_length]

        # Random segment
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

#####################################################################
# 5. Post-Processing Improvements
#####################################################################
def adaptive_noise_gate(audio, factor=1.2):
    """
    Zero out samples whose amplitude is below factor * mean amplitude.
    """
    threshold = factor * np.mean(np.abs(audio))
    audio_gated = np.where(np.abs(audio) < threshold, 0, audio)
    return audio_gated.astype(np.float32)

#####################################################################
# Training Loop with LR scheduler & optional STFT-based loss
#####################################################################


def train_model(model, dataloader, val_dataloader, epochs, device, learning_rate=1e-5,
                use_stft_loss=False):
    """
    :param use_stft_loss: If True, combines MSELoss with a multi-resolution STFT loss.
    """
    # Base criterion (MSE or L1).
    criterion = nn.MSELoss()

    # Optional advanced STFT-based loss
    stft_criterion = MultiResolutionSTFTLoss() if use_stft_loss else None

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add a learning rate scheduler (ReduceLROnPlateau example)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Keep your original grad scaler
    scaler = torch.amp.GradScaler(
        "cuda",  # Mixed precision training
        init_scale=2 ** 10,
        growth_factor=1.75,
        backoff_factor=0.75,
        growth_interval=150,
    )

    model.to(device)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            low_quality, high_quality = zip(*batch)
            low_quality = torch.stack(low_quality).to(device)
            high_quality = torch.stack(high_quality).to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(low_quality)
                
                # Base MSE loss
                loss = criterion(outputs, high_quality)

                # Optionally combine with STFT-based loss
                if stft_criterion is not None:
                    stft_loss_val = stft_criterion(outputs, high_quality)
                    # Combine them, e.g. 50/50 weighting
                    loss = 0.5 * loss + 0.5 * stft_loss_val

            scaler.scale(loss).backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.95)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                low_quality, high_quality = zip(*batch)
                low_quality = torch.stack(low_quality).to(device)
                high_quality = torch.stack(high_quality).to(device)

                with torch.amp.autocast("cuda"):
                    outputs = model(low_quality)
                    # Validation criterion
                    batch_loss = criterion(outputs, high_quality)
                    if stft_criterion is not None:
                        batch_loss = 0.5 * batch_loss + 0.5 * stft_criterion(outputs, high_quality)

                val_loss += batch_loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}")

        # Step the scheduler
        scheduler.step(avg_val_loss)
    

    return train_loss_history, val_loss_history


########################################
# Model Control
########################################
def model_ctrl(
                data_folder=r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\trainning_data_b1",
                b_size=8,
                t_frac=0.8,
                lr=1e-6,
                epoch_steps=32,
                samp_rate=96000,
                segment_len=0.25,
                use_stft_loss=False,
                use_augmentation=False

    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset
    dataset = PairedAudioDataset(
        folder_path=data_folder,
        sample_rate=samp_rate,
        segment_length=segment_len,
        apply_augmentation=use_augmentation
    )
    print(f"Total data pairs: {len(dataset)}")

    # Train/Val Split
    train_size = int(t_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # DataLoaders
    optimal_batch_size = min(b_size, len(train_dataset), len(val_dataset))
    # Using multiprocessing data loading (num_workers>0) and pin_memory=True for possible speed gains
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=optimal_batch_size,
        shuffle=False,
        num_workers=3,  # Increase if system resources allow
        pin_memory=False,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=optimal_batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=False,
        collate_fn=collate_fn
    )

    # Initialize and Train the Model
    model = AudioEnhancementModel()

    loss_t_data, loss_v_data = train_model(
        model,
        train_dataloader,
        val_dataloader,
        epochs=epoch_steps,
        device=device,
        learning_rate=lr,
        use_stft_loss=use_stft_loss
    )

    # Plot Training and Validation Loss
    epochs_range = range(len(loss_t_data))

    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, loss_t_data, label="Training Loss")
    plt.plot(epochs_range, loss_v_data, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    # Save Final Model and Hyperparameters
    model_path = r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\models\04.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "hyperparameters": {
            "batch_size": b_size,
            "learning_rate": lr,
            "sample_rate": samp_rate,
            "segment_length": segment_len
        }
    }, model_path)
    print(f"Model + hyperparams saved to {model_path}")

# Execute the Control Function (example)
if __name__ == "__main__":
    model_ctrl(
        data_folder=r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\trainning_data_b1",
        b_size=1,
        t_frac=0.75,
        lr=1e-6,
        epoch_steps=1,
        samp_rate=sr,
        segment_len=0.25,
        use_stft_loss=True,
        use_augmentation=True
    )
