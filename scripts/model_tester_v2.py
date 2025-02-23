import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

########################################
# Updated Model Class (from trainner_v6)
########################################
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

########################################
# Optional: Post-processing functions
########################################
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
    Apply noise suppression to an audio signal using spectral subtraction.
    """
    stft = librosa.stft(audio, n_fft=fft_size, win_length=fft_win_size)
    magnitude, phase = librosa.magphase(stft)

    # Estimate noise from quiet portions
    noise_estimate = np.median(magnitude, axis=1, keepdims=True)

    # Spectral subtraction
    magnitude_denoised = np.maximum(magnitude - noise_reduction_factor * noise_estimate, 0)
    stft_denoised = magnitude_denoised * phase
    audio_denoised = librosa.istft(stft_denoised)

    return audio_denoised

def butter_bandpass(lowcut, highcut, sample_rate:int, order=3):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, sample_rate:int):
    b, a = butter_bandpass(lowcut, highcut, sample_rate)
    return lfilter(b, a, data)

def volume_eq_bandpass(audio, sr, eq_settings=None, low_cutoff=100, high_cutoff=15000):
    """
    Apply adjustable EQ and bandpass filter.
    """
    if eq_settings is None:
        eq_settings = {
            100: 1.0,
            500: 1.0,
            1000: 1.0,
            5000: 1.0,
            10000: 1.0
        }

    # Bandpass first
    audio = bandpass_filter(audio, low_cutoff, high_cutoff, sr)

    # Then frequency-based gain
    frequencies = np.array(list(eq_settings.keys()))
    gains = np.array(list(eq_settings.values()))
    fft_data = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sr)

    for freq, gain in zip(frequencies, gains):
        # Simple example: apply gain to ±50 Hz around each target freq
        fft_data[(freqs >= freq - 50) & (freqs <= freq + 50)] *= gain

    audio_eq = np.fft.irfft(fft_data, n=len(audio))
    return audio_eq

########################################
# Main Tester
########################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to the new model checkpoint from trainner_v6
    model_path = r"models\03.pth" ################################################################################### SET ACTUAL MODEL PATH

    # Load Model and Hyperparameters
    checkpoint = torch.load(model_path, map_location=device)
    hyperparameters = checkpoint['hyperparameters']
    print("Loaded Hyperparameters:", hyperparameters)

    # Instantiate the updated model architecture
    model = AudioEnhancementModel().to(device)
    #print("Checkpoint keys:", checkpoint['model_state_dict'].keys())
    #print("Model keys:", model.state_dict().keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Model loaded successfully.")
    print("Generating audio...")

    # Example: input audio path
    audio_path = r"input_data\Savoy & Bright Lights - The Wolf (Savoy Live Version).wav" ############################## SET ACTUAL AUDIO PATH
    low_audio, sr = librosa.load(audio_path, sr=hyperparameters['sample_rate'], mono=True)

    # Convert to tensor and run inference
    low_tensor = torch.tensor(low_audio, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        enhanced_audio = model(low_tensor).cpu().numpy().squeeze()

    # Post-processing
    enhanced_audio = noise_gate(enhanced_audio, threshold=0.02)

    # Normalize to [-1,1]
    max_val = max(abs(enhanced_audio.max()), abs(enhanced_audio.min()))
    enhanced_audio = enhanced_audio / (max_val + 1e-10)

    # Optional bandpass/EQ
    eq_settings = {
        100:0.75,
        500:0.85,
        1000:0.9,
        3000:0.9,
        5000:0.9,
        10000:0.9,
        15000:0.9,
        20000:0.8
    }
    enhanced_audio = volume_eq_bandpass(enhanced_audio, sr, eq_settings, 20, 17000)

    # Save
    output_path = r"output_data\01.wav" ######################################## SET DESIRED AUDIO OUTPUT PATH (include file name and .wav)
    sf.write(output_path, enhanced_audio, sr)
    print(f"Enhanced audio saved to {output_path}")
