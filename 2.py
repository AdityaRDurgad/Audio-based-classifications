import os
import librosa
import numpy as np

TARGET_SR = 16000        # 16 kHz
TARGET_DURATION = 5      # 5 Seconds
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION  # 80,000 total values

def standardize_single_audio(file_path):
    """Loads, resamples to 16kHz, and pads/truncates to 5 seconds."""
    try:
        # Load clip and resample
        audio, sr = librosa.load(file_path, sr=TARGET_SR)
        
        # Pad with zeros if shorter than 5 seconds
        if len(audio) < TARGET_SAMPLES:
            padding = TARGET_SAMPLES - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        # Truncate if longer than 5 seconds
        else:
            audio = audio[:TARGET_SAMPLES]
            
        return audio
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def batch_process_audio(df, audio_dir):
    audio_signals = []
    valid_rows = []
    
    for idx, row in df.iterrows():
        full_path = os.path.join(audio_dir, row['path'])
        
        # Add .mp3 extension if not present in manifest
        if not full_path.endswith('.mp3') and not full_path.endswith('.wav'):
            full_path += '.mp3'
            
        processed_signal = standardize_single_audio(full_path)
        
        if processed_signal is not None:
            audio_signals.append(processed_signal)
            valid_rows.append(row)
            
    # Convert list to 2D NumPy Array [Number of Samples, 80000]
    return np.array(audio_signals), pd.DataFrame(valid_rows).reset_index(drop=True)

standardize_single_audio(file_path=r'D:\SnS_project\data\raw_audio')

# 1. First, run Step 2.1 to get the balanced DataFrame
balanced_df = clean_and_balance_metadata(
    tsv_path=r'D:\SnS_project\data\metadata\validated.tsv', 
    output_csv_path=r'D:\SnS_project\processed_data\balanced_metadata.csv'
)

# 2. Define the path to the directory containing your audio clips
audio_dir = r'D:\SnS_project\data\raw_audio'  # Update if your clips are in \data\clips

# 3. Call batch_process_audio (Step 2.2)
print("Starting audio standardization (16 kHz, 5 seconds)...")
audio_signals, valid_metadata_df = batch_process_audio(
    df=balanced_df, 
    audio_dir=audio_dir
)

# 4. Save the preprocessed outputs for Members 2, 3, and 4
os.makedirs(r'D:\SnS_project\processed_data', exist_ok=True)

# Save the standardized 2D numpy array [Number of Samples, 80000]
np.save(r'D:\SnS_project\processed_data\standardized_audio_signals.npy', audio_signals)

# Save the aligned metadata manifest CSV
valid_metadata_df.to_csv(r'D:\SnS_project\processed_data\valid_audio_metadata.csv', index=False)

print(f"Done! Standardized {len(audio_signals)} audio files.")
print(f"Output Matrix Shape: {audio_signals.shape}")
