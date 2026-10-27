import numpy as np
import scipy.sparse
from sklearn.neural_network import MLPClassifier  # Or use PyTorch/TensorFlow if preferred for DNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("==================================================")
print("MEMBER 3: MODEL TRAINING PIPELINE (Gender & Language)")
print("==================================================")

# =====================================================================
# PART 1: GENDER CLASSIFICATION (Audio Spectral DNN / MLP)
# =====================================================================
print("\n--- Training Gender Classification Model ---")

# Load bundled features from Member 2
data = np.load(r'D:\SnS_project\processed_data\final_features_for_models.npz', allow_pickle=True)
X_spec_train = data['X_spec_train']
X_spec_val = data['X_spec_val']
X_spec_test = data['X_spec_test']
y_g_train = data['y_g_train']
y_g_val = data['y_g_val']
y_g_test = data['y_g_test']

# Initialize and train a Multi-Layer Perceptron (DNN classifier)
gender_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)
gender_model.fit(X_spec_train, y_g_train)

# Evaluate on Validation set
val_preds = gender_model.predict(X_spec_val)
print(f"Gender Model Validation Accuracy: {accuracy_score(y_g_val, val_preds):.4f}")

# Save trained gender model
with open(r'D:\SnS_project\processed_data\gender_model.pkl', 'wb') as f:
    pickle.dump(gender_model, f)
print("Gender model saved successfully!")


# =====================================================================
# PART 2: LANGUAGE CLASSIFICATION (Text N-gram Classifier)
# =====================================================================
print("\n--- Training Language Classification Model ---")

# Load vectorized sparse text matrices
X_text_train = scipy.sparse.load_npz(r'D:\SnS_project\processed_data\text_vec_train.npz')
X_text_val = scipy.sparse.load_npz(r'D:\SnS_project\processed_data\text_vec_val.npz')

# Load language target labels from Member 1's text splits
text_splits = np.load(r'D:\SnS_project\processed_data\text_splits.npz', allow_pickle=True)
y_lang_train = text_splits['y_train']
y_lang_val = text_splits['y_val']

# Initialize and train a Logistic Regression classifier for high-dimensional sparse text
lang_model = LogisticRegression(max_iter=1000, random_state=42)
lang_model.fit(X_text_train, y_lang_train)

# Evaluate on Validation set
lang_val_preds = lang_model.predict(X_text_val)
print(f"Language Model Validation Accuracy: {accuracy_score(y_lang_val, lang_val_preds):.4f}")

# Save trained language model
with open(r'D:\SnS_project\processed_data\language_model.pkl', 'wb') as f:
    pickle.dump(lang_model, f)
print("Language model saved successfully!")