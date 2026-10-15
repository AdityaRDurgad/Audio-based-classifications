# Multimodal Intelligence System: Feature Extraction Pipeline

A collaborative machine learning pipeline built to process and extract acoustic and textual features for downstream gender, language, and age classification models.

---

## 👥 Project Team & Workflow
* **Member 1**: Raw data acquisition, cleaning, text null-handling, and stratified splitting.
* **Member 2 (You)**: Acoustic feature engineering (MFCCs, Mel-spectrograms, Pitch/Tempo) & Text Vectorization (N-grams).
* **Member 3**: Gender (DNN) & Language (Naïve Bayes / Classifier) Model Training.
* **Member 4**: Age Estimation Pipeline & System GUI Integration.

---

## 🔬 Signals & Systems Theoretical Foundations
Our feature extraction phase bridges classical signal processing with modern machine learning:
1. **Nyquist-Shannon Sampling**: Audio signals are standardized at $16\text{ kHz}$ sampling rate ($f_s$), ensuring frequency representation up to $8\text{ kHz}$ without aliasing.
2. **STFT & Windowing**: Short-Time Fourier Transform applies sliding analysis windows to convert non-stationary audio signals into the time-frequency domain.
3. **Mel-Scale Warping**: Non-linear frequency scaling that mimics the human ear's logarithmic perception of pitch, extracting 128 Mel-filterbank energies and 20 Mel-Frequency Cepstral Coefficients (MFCCs).
4. **Ergodic Data Splitting**: Preserving stationary statistical properties across training, validation, and test splits to prevent data leakage.

---

## 📂 Directory Structure
```text
D:\SnS_project\
│
├── data/                             # Raw datasets (Managed by Member 1)
├── processed_data/                   # Feature matrices & serialized models
│   ├── final_features_for_models.npz # Bundled X features and y target labels
│   ├── text_vec_train.npz            # Vectorized sparse training text
│   └── text_vectorizer.pkl           # Saved CountVectorizer model
│
├── 1.py                              # Preprocessing script (Member 1)
├── feature_extraction.py             # Feature extraction script (Member 2)
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation
