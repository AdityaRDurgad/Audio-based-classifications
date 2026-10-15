import pandas as pd
import os
import librosa
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import train_test_split


def clean_and_balance_metadata(tsv_path, output_csv_path):
    # 1. Load TSV metadata
    df = pd.read_csv(tsv_path, sep='\t')
    
    # 2. Select relevant columns and drop rows missing gender or age
    df = df[['path', 'gender', 'age']].dropna()
    
    # 3. Filter out ambiguous or invalid entries
    df = df[df['gender'].isin(['male', 'female'])]
    
    # 4. Find the minimum count to balance male and female classes evenly
    class_counts = df['gender'].value_counts()
    min_samples = class_counts.min()
    
    male_df = df[df['gender'] == 'male'].sample(n=min_samples, random_state=42)
    female_df = df[df['gender'] == 'female'].sample(n=min_samples, random_state=42)
    
    # 5. Combine and shuffle
    balanced_df = pd.concat([male_df, female_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save clean metadata manifest
    balanced_df.to_csv(output_csv_path, index=False)
    print(f"Success: Metadata balanced with {min_samples} Male and {min_samples} Female records.")
    return balanced_df

balanced_df = clean_and_balance_metadata(tsv_path=r'D:\SnS_project\data\metadata\validated.tsv', output_csv_path=r'D:\SnS_project\processed_data\balanced_audio_metadata.csv')

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


# 1. First, run Step 2.1 to get the balanced DataFrame
balanced_df = clean_and_balance_metadata(tsv_path=r'D:\SnS_project\data\metadata\validated.tsv', output_csv_path=r'D:\SnS_project\processed_data\balanced_metadata.csv')

# 2. Define the path to the directory containing your audio clips
audio_dir = r'D:\SnS_project\data\raw_audio'  # Update if your clips are in \data\clips

# 3. Call batch_process_audio (Step 2.2)
print("Starting audio standardization (16 kHz, 5 seconds)...")
audio_signals, valid_metadata_df = batch_process_audio(df=balanced_df, audio_dir=audio_dir)

# 4. Save the preprocessed outputs for Members 2, 3, and 4
os.makedirs(r'D:\SnS_project\processed_data', exist_ok=True)

# Save the standardized 2D numpy array [Number of Samples, 80000]
np.save(r'D:\SnS_project\processed_data\standardized_audio_signals.npy', audio_signals)

# Save the aligned metadata manifest CSV
valid_metadata_df.to_csv(r'D:\SnS_project\processed_data\valid_audio_metadata.csv', index=False)

print(f"Done! Standardized {len(audio_signals)} audio files.")
print(f"Output Matrix Shape: {audio_signals.shape}")


def clean_raw_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Convert text to lowercase
    text = text.lower()
    
    # 2. Remove URLs and links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove numbers and digits
    text = re.sub(r'\d+', '', text)
    
    # 4. Remove brackets, punctuation, and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 2. Load your raw language detection text dataset
raw_text_path = r'D:\SnS_project\data\raw_text\Language_Detection.csv'  # Update with your file name
df_text = pd.read_csv(raw_text_path)

# 3. Apply the cleaning function to your text column (e.g., 'Text')
print("Cleaning text data...")
df_text['clean_text'] = df_text['Text'].apply(clean_raw_text)

# 4. Save the cleaned DataFrame to the processed_data folder
output_path = r'D:\SnS_project\processed_data\cleaned_text_dataset.csv'
df_text.to_csv(output_path, index=False)

print("Text cleaning complete!")
print(df_text[['Text', 'clean_text']].head())


def process_text_dataset(csv_path, text_col='Text', label_col='Language'):
    df = pd.read_csv(csv_path)
    
    # Drop empty rows
    df = df.dropna(subset=[text_col, label_col])
    
    # Apply regex cleaning
    df['clean_text'] = df[text_col].apply(clean_raw_text)
    
    # Encode categorical language labels
    label_encoder = LabelEncoder()
    df['encoded_language'] = label_encoder.fit_transform(df[label_col])
    
    # Save the label encoder mapping for the team to decode predictions later
    with open('processed_data/language_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
        
    print(f"Languages encoded: {list(label_encoder.classes_)}")
    return df

process_text_dataset(r'D:\SnS_project\data\raw_text\Language_Detection.csv', text_col='Text', label_col='Language')


def create_stratified_splits(X, y, test_size=0.15, val_size=0.15, random_seed=42):
    # Convert PyArrow / Pandas inputs to standard NumPy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    # 1. First split: Separate test set from train + validation set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, stratify=y)

    # 2. Adjust val_size relative to the remaining train_val portion
    adjusted_val_size = val_size / (1.0 - test_size)

    # 3. Second split: Separate train set from validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=adjusted_val_size, random_state=random_seed, stratify=y_train_val)

    return X_train, X_val, X_test, y_train, y_val, y_test


# 1. Load your standardized audio signals and metadata
audio_signals = np.load(r'D:\SnS_project\processed_data\standardized_audio_signals.npy')
metadata_df = pd.read_csv(r'D:\SnS_project\processed_data\valid_audio_metadata.csv')

# 2. Extract feature matrix X and target labels y
X_audio = audio_signals
y_gender = metadata_df['gender'].values  # Target labels for gender classification
y_age = metadata_df['age'].values        # Target labels for age classification

# 3. Call the stratified split function for Gender
X_train_g, X_val_g, X_test_g, y_train_g, y_val_g, y_test_g = create_stratified_splits(X=X_audio, y=y_gender, test_size=0.15, val_size=0.15, random_seed=42)

# 4. Save the split arrays into a single compressed file for Members 2, 3, and 4
np.savez_compressed(r'D:\SnS_project\processed_data\audio_splits.npz',X_train=X_train_g, X_val=X_val_g, X_test=X_test_g, y_train_g=y_train_g, y_val_g=y_val_g, y_test_g=y_test_g)

# Convert y_age to a Pandas Series to count frequencies easily
y_age_series = pd.Series(y_age)

# Keep only classes that have at least 3 samples
valid_classes = y_age_series.value_counts()[y_age_series.value_counts() >= 3].index

# Create a boolean mask to filter both X and y
mask = y_age_series.isin(valid_classes).values

X_audio_filtered = np.asarray(X_audio)[mask]
y_age_filtered = np.asarray(y_age)[mask]

print(f"Removed {len(y_age) - len(y_age_filtered)} samples from rare classes.")

# Now run stratified split on the filtered dataset
X_train_a, X_val_a, X_test_a, y_train_a, y_val_a, y_test_a = create_stratified_splits(X=X_audio_filtered, y=y_age_filtered, test_size=0.15, val_size=0.15, random_seed=42)

np.savez_compressed(r'D:\SnS_project\processed_data\audio_splits.npz', X_train=X_train_a, X_val=X_val_a, X_test=X_test_a, y_train_g=y_train_a, y_val_g=y_val_a, y_test_g=y_test_a)

print("Audio splits created and saved successfully!")


# 1. Load the preprocessed text dataset from Step 3
df_text = pd.read_csv(r'D:\SnS_project\processed_data\cleaned_text_dataset.csv')

# 1. Initialize the LabelEncoder
le = LabelEncoder()

# 2. Create the 'encoded_language' column from your raw 'language' column
# Print existing columns to inspect if needed
print("Available columns in df_text:", df_text.columns.tolist())

# List of common column names for language metadata
candidate_cols = ['language', 'locale', 'lang', 'accent']

# Match against candidate columns (ignoring case and whitespace)
target_col = next(
    (col for col in df_text.columns if col.strip().lower() in candidate_cols), 
    None
)

if target_col is not None:
    df_text['encoded_language'] = le.fit_transform(df_text[target_col])
    y_language = df_text['encoded_language'].values
    print(f"Encoded language targets using column: '{target_col}'")
else:
    raise KeyError(
        f"Could not find a language column. Available columns are: {df_text.columns.tolist()}"
    )

# 3. Now extract y_language safely
# 2. Extract feature Series X and label Series y
print("Actual columns in df_text:", df_text.columns.tolist())
X_text = df_text['clean_text'].values
y_language = df_text['encoded_language'].values

# 3. Call the stratified split function for Language Detection
X_text_train, X_text_val, X_text_test, y_lang_train, y_lang_val, y_lang_test = create_stratified_splits(X=X_text, y=y_language, test_size=0.15, val_size=0.15, random_seed=42)

# 4. Save the text splits to CSV files or a single NumPy archive
np.savez_compressed(r'D:\SnS_project\processed_data\text_splits.npz', X_train=X_text_train, X_val=X_text_val, X_test=X_text_test, y_train=y_lang_train, y_val=y_lang_val, y_test=y_lang_test)

print("Text splits created and saved successfully!")


# Saving audio array binaries
np.savez_compressed('processed_data/audio_processed_splits.npz', X_train=X_train_g, X_val=X_val_g, X_test=X_test_g, y_gender_train=y_train_g, y_gender_val=y_val_g, y_gender_test=y_test_g, y_age_train=y_train_a, y_age_val=y_val_a, y_age_test=y_test_a)

# Saving cleaned text CSV
df_text.to_csv('processed_data/cleaned_text_dataset.csv', index=False)