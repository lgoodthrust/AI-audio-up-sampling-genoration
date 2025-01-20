import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf  # For saving audio

# Step 2: Define the Model
class AudioEnhancementModel(nn.Module):
    def __init__(self):
        super(AudioEnhancementModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(16),  # Batch normalization
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
    
def noise_gate(audio, threshold=0.01):
    """
    Apply a noise gate to suppress low-amplitude noise.
    :param audio: Enhanced audio signal.
    :param threshold: Amplitude below which noise is suppressed.
    :return: Noise-gated audio.
    """
    audio_gated = np.where(np.abs(audio) < threshold, 0, audio)
    return audio_gated

def noise_suppression(audio, sr, noise_reduction_factor=0.25, fft_size=2048, fft_win_size=2048):
    """
    Apply noise suppression to an audio signal.
    :param audio: Input audio signal.
    :param sr: Sampling rate.
    :param noise_reduction_factor: Strength of noise reduction (0.0-1.0).
    :return: Noise-suppressed audio.
    """
    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio, n_fft=fft_size, win_length=fft_win_size)
    magnitude, phase = librosa.magphase(stft)

    # Estimate noise from the quiet parts of the signal
    noise_estimate = np.median(magnitude, axis=1, keepdims=True)

    # Apply spectral subtraction
    magnitude_denoised = np.maximum(magnitude - noise_reduction_factor * noise_estimate, 0)

    # Reconstruct the denoised audio
    stft_denoised = magnitude_denoised * phase
    audio_denoised = librosa.istft(stft_denoised)
    return audio_denoised


# Step 5: Initialize and Load the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the model checkpoint
model_path = r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\models\01.pth"

# Load Model and Hyperparameters
checkpoint = torch.load(model_path, map_location=device)

# Extract hyperparameters
hyperparameters = checkpoint['hyperparameters']
print("Loaded Hyperparameters:", hyperparameters)

# Reconstruct the model
model = AudioEnhancementModel().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully.")
print("generating audio...")

# Path to the input audio
audio_path = r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\input_data\Savoy & Bright Lights - The Wolf (Savoy Live Version)_2.wav"

# Load and process the low-quality audio
low_audio, sr = librosa.load(audio_path, sr=hyperparameters['sample_rate'], mono=True)

# Apply noise suppression
low_audio_denoised = noise_suppression(low_audio, sr, fft_size=8192, fft_win_size=8192)

# Convert to tensor
low_tensor = torch.tensor(low_audio_denoised, dtype=torch.float32).unsqueeze(0).to(device)

# Enhance the audio
with torch.no_grad():
    enhanced_audio = model(low_tensor).cpu().numpy().squeeze()

# Post-process: Noise gate and normalization
enhanced_audio = noise_gate(enhanced_audio, threshold=0.02)
max_val = max(abs(enhanced_audio.max()), abs(enhanced_audio.min()))
enhanced_audio = enhanced_audio / max_val  # Normalize to [-1.0, 1.0]

# Save the enhanced audio
output_path = r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\output_data\05.wav"
sf.write(output_path, enhanced_audio, sr)
print(f"Enhanced and noise-suppressed audio saved to {output_path}")
