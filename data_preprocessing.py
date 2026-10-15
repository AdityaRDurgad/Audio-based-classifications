import os
import librosa
import numpy as np
import pandas as pd

TARGET_SR = 16000  # 16 kHz
TARGET_DURATION = 5  # 5 seconds
TARGET_LENGTH = TARGET_SR * TARGET_DURATION  # 80,000 samples

def preprocess_audio_file(file_path):
    """Loads, resamples, and pads/truncates audio to a fixed length."""
    try:
        # Load audio and resample to 16 kHz
        audio, sr = librosa.load(file_path, sr=TARGET_SR)
        
        # Pad or truncate to exactly 5 seconds
        if len(audio) < TARGET_LENGTH:
            # Pad with zeros if shorter than 5 seconds
            pad_width = TARGET_LENGTH - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
        else:
            # Truncate if longer than 5 seconds
            audio = audio[:TARGET_LENGTH]
            
        return audio
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_audio_dataset(df, audio_dir):
    """Iterates through dataframe and standardizes audio files."""
    processed_audio_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        file_path = os.path.join(audio_dir, row['path'])
        audio_data = preprocess_audio_file(file_path)
        
        if audio_data is not None:
            processed_audio_list.append(audio_data)
            valid_indices.append(idx)
            
    filtered_df = df.loc[valid_indices].reset_index(drop=True)
    return np.array(processed_audio_list), filtered_df



def load_and_balance_common_voice(tsv_path):
    """Parses Mozilla Common Voice TSV/CSV metadata and balances gender classes."""
    df = pd.read_csv(tsv_path, sep='\t')
    
    # 1. Keep relevant columns and remove missing values
    df = df[['path', 'gender', 'age']].dropna()
    
    # 2. Filter out ambiguous gender labels
    df = df[df['gender'].isin(['male', 'female'])]
    
    # 3. Balance Male and Female sample counts
    min_count = df['gender'].value_counts().min()
    male_df = df[df['gender'] == 'male'].sample(min_count, random_state=42)
    female_df = df[df['gender'] == 'female'].sample(min_count, random_state=42)
    
    balanced_df = pd.concat([male_df, female_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset Balanced: {min_count} Male and {min_count} Female samples.")
    return balanced_df



import re
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    """Removes special characters, numbers, and converts text to lowercase."""
    text = re.sub(r'[\(\)\[\]\{\}\\\/]', '', text)  # Remove brackets
    text = re.sub(r'[0-9]+', '', text)             # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)            # Remove punctuation
    text = text.lower().strip()                   # Convert to lowercase
    return text

def preprocess_text_dataset(df, text_col='Text', label_col='Language'):
    """Cleans text dataset and encodes categorical language labels."""
    df[f'clean_{text_col}'] = df[text_col].apply(clean_text)
    
    # Encode target labels
    encoder = LabelEncoder()
    df['encoded_label'] = encoder.fit_transform(df[label_col])
    
    return df, encoder



from sklearn.model_selection import train_test_split

def create_data_splits(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Splits features and labels into Train, Validation, and Test sets."""
    # First split into Train+Val and Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Calculate adjusted validation size
    adjusted_val_size = val_size / (1.0 - test_size)
    
    # Split Train+Val into Train and Val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, random_state=random_state, stratify=y_train_val
    )
    
    print(f"Data Splits Created: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test