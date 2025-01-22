import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

# Model Class
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

def noise_suppression(audio, noise_reduction_factor=0.75, fft_size=2048, fft_win_size=2048):
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

# Butterpass filter
def butter_bandpass(lowcut, highcut, sample_rate:int, order=3) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Bandpass filter
def bandpass_filter(data, lowcut, highcut, sample_rate:int) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, sample_rate)
    return lfilter(b, a, data)

def volume_eq_bandpass(audio, sr, eq_settings=None, low_cutoff=100, high_cutoff=15000):
    """
    Apply an adjustable equalizer and bandpass filter to the audio signal.
    :param audio: Input audio signal.
    :param sr: Sampling rate of the audio.
    :param eq_settings: Dictionary containing frequency bands and their respective gain values.
    :param low_cutoff: Low cutoff frequency for the bandpass filter.
    :param high_cutoff: High cutoff frequency for the bandpass filter.
    :return: Processed audio.
    """
    if eq_settings is None:
        eq_settings = {
            100: 1.0,
            500: 1.0,
            1000: 1.0,
            5000: 1.0,
            10000: 1.0
        }

    # Apply bandpass filter
    audio = bandpass_filter(audio, low_cutoff, high_cutoff, sr)

    # Apply equalizer
    frequencies = np.array(list(eq_settings.keys()))
    gains = np.array(list(eq_settings.values()))
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sr)

    for freq, gain in zip(frequencies, gains):
        fft[(freqs >= freq - 50) & (freqs <= freq + 50)] *= gain

    audio_eq = np.fft.irfft(fft, n=len(audio))
    return audio_eq

# Step 5: Initialize and Load the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the model checkpoint
model_path = r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\models\02.pth"

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
audio_path = r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\input_data\Savoy & Bright Lights - The Wolf (Savoy Live Version).wav"

# Load and process the low-quality audio
low_audio, sr = librosa.load(audio_path, sr=hyperparameters['sample_rate'], mono=True)

# Convert to tensor
low_tensor = torch.tensor(low_audio, dtype=torch.float32).unsqueeze(0).to(device)

# Enhance the audio
with torch.no_grad():
    enhanced_audio = model(low_tensor).cpu().numpy().squeeze()

# Post-process: Noise gate and normalization
enhanced_audio = noise_gate(enhanced_audio, threshold=0.08)
max_val = max(abs(enhanced_audio.max()), abs(enhanced_audio.min()))
enhanced_audio = enhanced_audio / max_val  # Normalize to [-1.0, 1.0]

enhanced_audio = volume_eq_bandpass(enhanced_audio, sr, {100:0.75, 500:0.85, 1000:0.9, 3000:0.9, 5000:0.9, 10000:0.9, 15000:0.9, 20000:0.8}, 20, 17000)

# Save the enhanced audio
output_path = r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\output_data\09.wav"
sf.write(output_path, enhanced_audio, sr)
print(f"Enhanced and noise-suppressed audio saved to {output_path}")
