import numpy as np
import os

folder = r'D:\SnS_project\processed_data'
for file in os.listdir(folder):
    if file.endswith('.npz'):
        path = os.path.join(folder, file)
        try:
            data = np.load(path, allow_pickle=True)
            print(f"File: {file}")
            print(f"  Keys: {list(data.keys())}")
        except Exception as e:
            print(f"File: {file} - Error: {e}")