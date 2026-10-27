import numpy as np
import scipy.sparse
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("==================================================")
print("MEMBER 3: OPTIMIZING & EVALUATING MODELS")
print("==================================================")

# =====================================================================
# 1. FIX & RETRAIN GENDER MODEL WITH FEATURE SCALING
# =====================================================================
print("\n--- Retraining Gender Classification Model with Scaling ---")
data = np.load(r'D:\SnS_project\processed_data\final_features_for_models.npz', allow_pickle=True)
X_spec_train = data['X_spec_train']
X_spec_val = data['X_spec_val']
X_spec_test = data['X_spec_test']
y_g_train = data['y_g_train']
y_g_val = data['y_g_val']
y_g_test = data['y_g_test']

# Scale features (Critical for Neural Networks / MLPs)
scaler = StandardScaler()
X_spec_train_scaled = scaler.fit_transform(X_spec_train)
X_spec_val_scaled = scaler.transform(X_spec_val)
X_spec_test_scaled = scaler.transform(X_spec_test)

# Train MLP with scaled data
gender_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)
gender_model.fit(X_spec_train_scaled, y_g_train)

val_preds = gender_model.predict(X_spec_val_scaled)
print(f"New Gender Model Validation Accuracy (with scaling): {accuracy_score(y_g_val, val_preds):.4f}")

# Save scaler and improved gender model
with open(r'D:\SnS_project\processed_data\gender_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open(r'D:\SnS_project\processed_data\gender_model.pkl', 'wb') as f:
    pickle.dump(gender_model, f)


# =====================================================================
# 2. FINAL TEST SET EVALUATION FOR BOTH MODELS
# =====================================================================
print("\n--- Evaluating on Test Set (Final Hand-off Metrics) ---")

# Gender Test Evaluation
gender_test_preds = gender_model.predict(X_spec_test_scaled)
print(f"Gender Model Test Accuracy: {accuracy_score(y_g_test, gender_test_preds):.4f}")

# Language Test Evaluation
X_text_test = scipy.sparse.load_npz(r'D:\SnS_project\processed_data\text_vec_test.npz')
text_splits = np.load(r'D:\SnS_project\processed_data\text_splits.npz', allow_pickle=True)
y_lang_test = text_splits['y_test']

with open(r'D:\SnS_project\processed_data\language_model.pkl', 'rb') as f:
    lang_model = pickle.load(f)

lang_test_preds = lang_model.predict(X_text_test)
print(f"Language Model Test Accuracy: {accuracy_score(y_lang_test, lang_test_preds):.4f}")

print("\nAll model optimization and test evaluations complete!")