import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("==================================================")
print("FIXING GENDER MODEL WITH TRUE MALE/FEMALE LABELS")
print("==================================================")

# 1. Load feature matrices from final_features_for_models.npz
data = np.load(r'D:\SnS_project\processed_data\final_features_for_models.npz', allow_pickle=True)
X_spec_train = data['X_spec_train']
X_spec_val = data['X_spec_val']
X_spec_test = data['X_spec_test']

# 2. Load true gender labels from audio_processed_splits.npz
audio_splits = np.load(r'D:\SnS_project\processed_data\audio_processed_splits.npz', allow_pickle=True)
y_gender_train_raw = audio_splits['y_gender_train']
y_gender_val_raw = audio_splits['y_gender_val']
y_gender_test_raw = audio_splits['y_gender_test']

print(f"Feature train shape: {X_spec_train.shape} | Label train shape: {y_gender_train_raw.shape}")
print(f"Feature val shape:   {X_spec_val.shape}   | Label val shape:   {y_gender_val_raw.shape}")

# 3. Handle any minor length discrepancy due to dropped audio files during feature extraction
def align_labels(X_features, y_labels):
    if len(X_features) != len(y_labels):
        print(f"Aligning lengths: Features ({len(X_features)}) vs Labels ({len(y_labels)})")
        return y_labels[:len(X_features)]
    return y_labels

y_g_train = align_labels(X_spec_train, y_gender_train_raw)
y_g_val = align_labels(X_spec_val, y_gender_val_raw)
y_g_test = align_labels(X_spec_test, y_gender_test_raw)

print(f"Unique true gender classes: {np.unique(y_g_train)}")

# 4. Scale spectral features
scaler = StandardScaler()
X_spec_train_scaled = scaler.fit_transform(X_spec_train)
X_spec_val_scaled = scaler.transform(X_spec_val)
X_spec_test_scaled = scaler.transform(X_spec_test)

# 5. Train MLP Classifier for Gender
gender_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)
gender_model.fit(X_spec_train_scaled, y_g_train)

# 6. Evaluate validation performance
val_preds = gender_model.predict(X_spec_val_scaled)
print(f"\nCorrected Gender Validation Accuracy: {accuracy_score(y_g_val, val_preds):.4f}")
print("\nClassification Report:")
print(classification_report(y_g_val, val_preds))

# 7. Save corrected model and scaler
with open(r'D:\SnS_project\processed_data\gender_model.pkl', 'wb') as f:
    pickle.dump(gender_model, f)
with open(r'D:\SnS_project\processed_data\gender_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nCorrected gender model saved successfully as gender_model.pkl!")