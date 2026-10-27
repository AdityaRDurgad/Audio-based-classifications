import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

print("==================================================")
print("FIXING GENDER MODEL (Using Aligned Split Labels)")
print("==================================================")

# 1. Load spectral features from final_features_for_models.npz
data = np.load(r'D:\SnS_project\processed_data\final_features_for_models.npz', allow_pickle=True)
X_spec_train = data['X_spec_train']
X_spec_val = data['X_spec_val']
X_spec_test = data['X_spec_test']

# 2. Load aligned gender labels from audio_splits.npz (matches feature count)
audio_splits = np.load(r'D:\SnS_project\processed_data\audio_splits.npz', allow_pickle=True)
y_g_train = audio_splits['y_train_g']
y_g_val = audio_splits['y_val_g']
y_g_test = audio_splits['y_test_g']

print(f"X_spec_val shape: {X_spec_val.shape}")
print(f"y_g_val shape: {y_g_val.shape}")
print(f"Unique gender classes: {np.unique(y_g_train)}")

# 3. Scale features
scaler = StandardScaler()
X_spec_train_scaled = scaler.fit_transform(X_spec_train)
X_spec_val_scaled = scaler.transform(X_spec_val)

# 4. Train the MLP Classifier with correct gender labels
gender_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)
gender_model.fit(X_spec_train_scaled, y_g_train)

# Evaluate validation accuracy
val_preds = gender_model.predict(X_spec_val_scaled)
print(f"Corrected Gender Model Validation Accuracy: {accuracy_score(y_g_val, val_preds):.4f}")

# 5. Save the fixed model and scaler
with open(r'D:\SnS_project\processed_data\gender_model.pkl', 'wb') as f:
    pickle.dump(gender_model, f)
with open(r'D:\SnS_project\processed_data\gender_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Gender model fixed and saved successfully!")