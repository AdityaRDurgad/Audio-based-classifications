import pickle
import numpy as np

# 1. Load the trained language model
with open(r'D:\SnS_project\processed_data\language_model.pkl', 'rb') as f:
    language_model = pickle.load(f)

print("Language Model Classes (Mapping order):")
for idx, class_name in enumerate(language_model.classes_):
    print(f"  Index {idx}: {class_name}")

# 2. Check text splits to see original string labels
try:
    text_splits = np.load(r'D:\SnS_project\processed_data\text_splits.npz', allow_pickle=True)
    if 'y_train' in text_splits:
        print(f"\nUnique original language labels in dataset: {np.unique(text_splits['y_train'])}")
except Exception as e:
    print(f"\nCould not load text_splits.npz: {e}")