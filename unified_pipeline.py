import pickle
import numpy as np
import scipy.sparse

print("==================================================")
print("MULTIMODAL SYSTEM: UNIFIED INFERENCE DEMO")
print("==================================================")

# 1. Load all saved models and transformers
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

print("All models and vectorizers loaded successfully into memory!")

# 2. Grab a sample test instance from the test splits
data = np.load(r'D:\SnS_project\processed_data\final_features_for_models.npz', allow_pickle=True)
sample_spec = data['X_spec_test'][0].reshape(1, -1)
sample_pitch = data['X_pitch_test'][0].reshape(1, -1)

# Gender Prediction
sample_spec_scaled = gender_scaler.transform(sample_spec)
pred_gender = gender_model.predict(sample_spec_scaled)[0]

# Age Category Prediction
pred_age_group = age_model.predict(sample_pitch)[0]

# Language Prediction
sample_text = ["This is a test transcript for multimodal language classification."]
sample_text_vec = text_vectorizer.transform(sample_text)
pred_language = language_model.predict(sample_text_vec)[0]

print("\n--- Live Multimodal Inference Results ---")
print(f"Predicted Gender    : {pred_gender}")
print(f"Predicted Age Group : {pred_age_group}")
print(f"Predicted Language  : {pred_language}")
print("==================================================")