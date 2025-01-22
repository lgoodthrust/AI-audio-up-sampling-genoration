import os
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

def convert_mp3_to_wav(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all mp3 files in the input folder
    mp3_files = [f for f in os.listdir(input_folder) if f.endswith('.mp3')]

    if not mp3_files:
        print("No MP3 files found in the input folder.")
        return

    counter = 1
    for mp3_file in mp3_files:
        # Get the file size in MB
        mp3_path = os.path.join(input_folder, mp3_file)
        file_size_mb = os.path.getsize(mp3_path) / (1024 * 1024)

        # Check file size constraints
        if file_size_mb < 1 or file_size_mb > 30:
            print(f"Skipping {mp3_file}: file size {file_size_mb:.2f} MB is out of the allowed range (1MB - 30MB).")
            continue

        try:
            # Load MP3 file
            audio = AudioSegment.from_mp3(mp3_path)
        except CouldntDecodeError:
            print(f"Skipping {mp3_file}: Could not decode file. It might be corrupted or unsupported.")
            continue

        # Generate output file names
        low_bitrate_name = f"{counter}_low.wav"
        high_bitrate_name = f"{counter}_high.wav"

        low_bitrate_path = os.path.join(output_folder, low_bitrate_name)
        high_bitrate_path = os.path.join(output_folder, high_bitrate_name)

        # Export audio with different bitrates
        audio.set_frame_rate(8000).export(low_bitrate_path, format="wav")
        audio.set_frame_rate(44100).export(high_bitrate_path, format="wav")

        print(f"Converted: {mp3_file} to {low_bitrate_name} and {high_bitrate_name}")

        counter += 1

    print("Conversion complete.")


if __name__ == "__main__":
    input_folder = r"D:\other\sound board"
    output_folder = r"D:\code stuff\AAA\py scripts\audio_AI\UPSCALING\reserved_training_data"

    convert_mp3_to_wav(input_folder, output_folder)
