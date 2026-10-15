import numpy as np
import pandas as pd
import librosa
import os
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# =====================================================================
# STEP 1: LOAD MEMBER 1'S PREPROCESSED DATA
# =====================================================================
print("Loading preprocessed data from Member 1...")
audio_data = np.load(r'D:\SnS_project\processed_data\audio_splits.npz', allow_pickle=True)
X_audio_train = audio_data['X_train']
X_audio_val = audio_data['X_val']
X_audio_test = audio_data['X_test']

text_data = np.load(r'D:\SnS_project\processed_data\text_splits.npz', allow_pickle=True)
X_text_train = text_data['X_train']
X_text_val = text_data['X_val']
X_text_test = text_data['X_test']

print("Data successfully loaded from Member 1!")

# =====================================================================
# STEP 2: EXTRACT SPECTRAL FEATURES (MFCCs & Mel-Spectrograms)
# =====================================================================
def extract_spectral_features(audio_matrix, sr=16000, n_mfcc=20):
    features_list = []
    for i, signal in enumerate(audio_matrix):
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        mfcc_std = np.std(mfccs.T, axis=0)
        
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        mel_mean = np.mean(mel_spec.T, axis=0)
        
        combined_features = np.hstack([mfcc_mean, mfcc_std, mel_mean])
        features_list.append(combined_features)
        
        if (i + 1) % 500 == 0:
            print(f"Extracted spectral features for {i + 1}/{len(audio_matrix)} signals...")
            
    return np.array(features_list)

print("Extracting spectral features for Train, Validation, and Test sets...")
X_spectral_train = extract_spectral_features(X_audio_train)
X_spectral_val = extract_spectral_features(X_audio_val)
X_spectral_test = extract_spectral_features(X_audio_test)

# =====================================================================
# STEP 3: EXTRACT PITCH & TEMPO FEATURES
# =====================================================================
def extract_pitch_and_tempo_features(audio_matrix, sr=16000):
    pitch_features = []
    for signal in audio_matrix:
        pitches, magnitudes = librosa.piptrack(y=signal, sr=sr)
        f0 = pitches[pitches > 0]
        
        if len(f0) > 0:
            f0_mean = np.mean(f0)
            f0_std = np.std(f0)
            f0_median = np.median(f0)
        else:
            f0_mean, f0_std, f0_median = 0.0, 0.0, 0.0
            
        tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)
        tempo_val = float(np.mean(tempo))
        
        pitch_features.append([f0_mean, f0_std, f0_median, tempo_val])
        
    return np.array(pitch_features)

print("Extracting pitch and tempo features for Age estimation...")
X_pitch_train = extract_pitch_and_tempo_features(X_audio_train)
X_pitch_val = extract_pitch_and_tempo_features(X_audio_val)
X_pitch_test = extract_pitch_and_tempo_features(X_audio_test)

# =====================================================================
# STEP 4: TEXT VECTORIZATION USING COUNT VECTORIZER
# =====================================================================
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)

print("Cleaning and Vectorizing text data into N-gram term matrices...")
X_text_train_clean = [str(text) if pd.notna(text) else "" for text in X_text_train]
X_text_val_clean = [str(text) if pd.notna(text) else "" for text in X_text_val]
X_text_test_clean = [str(text) if pd.notna(text) else "" for text in X_text_test]

X_text_vec_train = vectorizer.fit_transform(X_text_train_clean)
X_text_vec_val = vectorizer.transform(X_text_val_clean)
X_text_vec_test = vectorizer.transform(X_text_test_clean)

print(f"Text Feature Matrix Shape: {X_text_vec_train.shape}")

os.makedirs(r'D:\SnS_project\processed_data', exist_ok=True)
with open(r'D:\SnS_project\processed_data\text_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# =====================================================================
# STEP 5: SAVE ALL EXTRACTED FEATURES & TARGET LABELS
# =====================================================================
print("Saving and bundling all feature files...")

# Save extracted audio features for Members 3 & 4
np.savez_compressed(
    r'D:\SnS_project\processed_data\extracted_audio_features.npz',
    X_spectral_train=X_spectral_train, X_spectral_val=X_spectral_val, X_spectral_test=X_spectral_test,
    X_pitch_train=X_pitch_train, X_pitch_val=X_pitch_val, X_pitch_test=X_pitch_test
)

# Save vectorized text sparse matrices
scipy.sparse.save_npz(r'D:\SnS_project\processed_data\text_vec_train.npz', X_text_vec_train)
scipy.sparse.save_npz(r'D:\SnS_project\processed_data\text_vec_val.npz', X_text_vec_val)
scipy.sparse.save_npz(r'D:\SnS_project\processed_data\text_vec_test.npz', X_text_vec_test)

# Load target labels safely with allow_pickle=True
labels_audio = np.load(r'D:\SnS_project\processed_data\audio_splits.npz', allow_pickle=True)
y_g_train, y_g_val, y_g_test = labels_audio['y_train_g'], labels_audio['y_val_g'], labels_audio['y_test_g']

labels_text = np.load(r'D:\SnS_project\processed_data\text_splits.npz', allow_pickle=True)
y_lang_train, y_lang_val, y_lang_test = labels_text['y_train'], labels_text['y_val'], labels_text['y_test']

# Save final feature + label bundle for model training
np.savez_compressed(
    r'D:\SnS_project\processed_data\final_features_for_models.npz',
    X_spec_train=X_spectral_train, X_spec_val=X_spectral_val, X_spec_test=X_spectral_test,
    X_pitch_train=X_pitch_train, X_pitch_val=X_pitch_val, X_pitch_test=X_pitch_test,
    y_g_train=y_g_train, y_g_val=y_g_val, y_g_test=y_g_test,
    y_lang_train=y_lang_train, y_lang_val=y_lang_val, y_lang_test=y_lang_test
)

print("All features extracted, bundled, and saved successfully!")