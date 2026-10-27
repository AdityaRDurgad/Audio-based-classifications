import os
import pickle
import numpy as np
import librosa

print("==================================================")
print("MULTIMODAL SYSTEM: CUSTOM LIVE PREDICTION (Updated)")
print("==================================================")

# 1. Load all saved models, scalers, vectorizers, and the language encoder
with open(r'D:\SnS_project\processed_data\gender_model.pkl', 'rb') as f:
    gender_model = pickle.load(f)
with open(r'D:\SnS_project\processed_data\gender_scaler.pkl', 'rb') as f:
    gender_scaler = pickle.load(f)
with open(r'D:\SnS_project\processed_data\language_model.pkl', 'rb') as f:
    language_model = pickle.load(f)
with open(r'D:\SnS_project\processed_data\text_vectorizer.pkl', 'rb') as f:
    text_vectorizer = pickle.load(f)
with open(r'D:\SnS_project\processed_data\age_model.pkl', 'rb') as f:
    age_model = pickle.load(f)
with open(r'D:\SnS_project\processed_data\language_label_encoder.pkl', 'rb') as f:
    lang_encoder = pickle.load(f)

print("All models and encoders loaded successfully!")

# =====================================================================
# 2. DEFINE YOUR INPUTS HERE
# =====================================================================
input_audio_path = r'D:\SnS_project\data\girll.wav'  # Change to your .wav file path
input_text = "This is a custom test sentence to check language classification."

# =====================================================================
# 3. FEATURE EXTRACTION FOR NEW AUDIO
# =====================================================================
def extract_custom_audio_features(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
    
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Extract Spectral Features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spec_features = np.hstack([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(mel, axis=1), np.std(mel, axis=1)
    ])
    
    # Extract Pitch Features
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_clean = f0[~np.isnan(f0)]
    pitch_features = np.array([
        np.mean(f0_clean) if len(f0_clean) > 0 else 0.0,
        np.std(f0_clean) if len(f0_clean) > 0 else 0.0,
        np.min(f0_clean) if len(f0_clean) > 0 else 0.0,
        np.max(f0_clean) if len(f0_clean) > 0 else 0.0
    ])
    
    return spec_features.reshape(1, -1), pitch_features.reshape(1, -1)

# =====================================================================
# 4. RUN PREDICTIONS & DECODE LABELS
# =====================================================================
try:
    print(f"\nProcessing audio file: {input_audio_path} ...")
    X_spec_custom, X_pitch_custom = extract_custom_audio_features(input_audio_path)
    
    # Match expected dimension for gender scaler
    expected_spec_dim = gender_scaler.mean_.shape[0]
    if X_spec_custom.shape[1] < expected_spec_dim:
        X_spec_custom = np.pad(X_spec_custom, ((0, 0), (0, expected_spec_dim - X_spec_custom.shape[1])))
    else:
        X_spec_custom = X_spec_custom[:, :expected_spec_dim]

    # Predict Gender & Age
    X_spec_scaled = gender_scaler.transform(X_spec_custom)
    pred_gender = gender_model.predict(X_spec_scaled)[0]
    pred_age = age_model.predict(X_pitch_custom)[0]

    # Predict Language and decode numeric ID back to string name
    text_vec = text_vectorizer.transform([input_text])
    pred_lang_idx = language_model.predict(text_vec)[0]
    pred_language = lang_encoder.inverse_transform([pred_lang_idx])[0]

    print("\n==================================================")
    print("           CUSTOM PREDICTION RESULTS              ")
    print("==================================================")
    print(f"Input Text         : \"{input_text}\"")
    print(f"Predicted Gender   : {pred_gender}")
    print(f"Predicted Age Group: {pred_age}")
    print(f"Predicted Language : {pred_language}")
    print("==================================================")

except Exception as e:
    print(f"\n[Note] Make sure a valid .wav file is provided.")
    print(f"Error details: {e}")