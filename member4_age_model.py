import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("==================================================")
print("MEMBER 4: AGE CLASSIFICATION MODEL TRAINING")
print("==================================================")

# 1. Load pitch features from Member 2's bundle
feature_data = np.load(r'D:\SnS_project\processed_data\final_features_for_models.npz', allow_pickle=True)
X_pitch_train = feature_data['X_pitch_train']
X_pitch_val = feature_data['X_pitch_val']
X_pitch_test = feature_data['X_pitch_test']

# 2. Load raw string age target labels directly
audio_splits = np.load(r'D:\SnS_project\processed_data\audio_processed_splits.npz', allow_pickle=True)
y_age_train = audio_splits['y_age_train']
y_age_val = audio_splits['y_age_val']
y_age_test = audio_splits['y_age_test']

print(f"Unique age categories found: {np.unique(y_age_train)}")

# 3. Initialize and train Random Forest Classifier
age_model = RandomForestClassifier(n_estimators=100, random_state=42)
age_model.fit(X_pitch_train, y_age_train)

# 4. Evaluate on Validation set
val_preds = age_model.predict(X_pitch_val)
acc = accuracy_score(y_age_val, val_preds)

print(f"\nAge Classification Validation Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_age_val, val_preds))

# 5. Save the trained age model
with open(r'D:\SnS_project\processed_data\age_model.pkl', 'wb') as f:
    pickle.dump(age_model, f)

print("\nAge classification model saved successfully as age_model.pkl!")