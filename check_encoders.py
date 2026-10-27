import os
import pickle

folder = r'D:\SnS_project\processed_data'
print("Scanning all .pkl files in processed_data for an encoder...")

for f in os.listdir(folder):
    if f.endswith('.pkl'):
        path = os.path.join(folder, f)
        try:
            with open(path, 'rb') as file:
                obj = pickle.load(file)
                print(f"\nFile: {f}")
                print(f"  Type: {type(obj)}")
                if hasattr(obj, 'classes_'):
                    print(f"  -> Contains classes_: {obj.classes_}")
                elif isinstance(obj, dict):
                    print(f"  -> Dictionary keys: {list(obj.keys())[:5]}")
        except Exception as e:
            print(f"\nFile: {f} - Could not load: {e}")