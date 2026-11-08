# BITS Pilani - Work Integrated Learning Programmes (WILP)
## M.Tech in Artificial Intelligence & Machine Learning (AIML)
### Course: Machine Learning S1-26_AIMLZG565 (Assignment - 2)

### Name : Rutaansh Nambiar
### BITS - ID : 2026AK05526
---

## a. Problem Statement
In the contemporary music streaming ecosystem, identifying the acoustic and sonic characteristics that propel a musical track into global virality is of paramount commercial and cultural value for record labels, digital streaming platforms (Spotify, Apple Music), and independent artists. 

The objective of this assignment is to develop an end-to-end Machine Learning classification pipeline that accurately classifies whether a given song will become **Viral** (`viral = 1`) or remain **Non-Viral** (`viral = 0`) based on its multi-dimensional acoustic and production features. The pipeline encompasses model training across **5 distinct machine learning classifiers**, rigorous evaluation over **6 standardized performance metrics**, serialization of trained models, and deployment as an interactive, zero-retraining web application using **Streamlit Community Cloud**.

---

## b. Dataset Description
The model is trained on the **Viral Songs Dataset**, a curated tabular dataset capturing audio engineering and distribution metrics across 10,000 song releases:

- **Instance Count:** 10,000 instances (substantially exceeds the assignment requirement of $\ge 500$ instances).
- **Feature Count:** 13 input features (12 numerical acoustic features + 1 categorical feature, satisfying the requirement of $\ge 12$ features).
- **Target Variable:** `viral` (Binary Classification: `0` = Non-Viral [63.95%, 6,395 samples], `1` = Viral [36.05%, 3,605 samples]).
- **Missing Values:** 0 missing values across all columns.
- **Train/Test Split:** 80% Training (8,000 instances) / 20% Stratified Holdout Test Set (2,000 instances) with fixed `random_state=42`.

### Feature Dictionary:
| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `genre` | Categorical | Musical genre (10 categories: Pop, Rock, Hip-Hop, Latin, Indie, Electronic, Classical, Jazz, R&B, Country) |
| `release_year` | Numerical | Year of track release (2009 – 2024) |
| `duration_sec` | Numerical | Total track length in seconds |
| `danceability` | Numerical | Suitability for dancing (0.0 to 1.0) |
| `energy` | Numerical | Perceptual measure of sonic intensity and activity (0.0 to 1.0) |
| `valence` | Numerical | Musical positiveness and mood conveying (0.0 to 1.0) |
| `tempo_bpm` | Numerical | Estimated track tempo in beats per minute |
| `acousticness` | Numerical | Confidence measure of acoustic instrumentation (0.0 to 1.0) |
| `instrumentalness` | Numerical | Likelihood that track contains no vocal content (0.0 to 1.0) |
| `speechiness` | Numerical | Presence of spoken words vs musical singing (0.0 to 1.0) |
| `liveness` | Numerical | Probability of audience presence / live recording (0.0 to 1.0) |
| `loudness_db` | Numerical | Overall track loudness in decibels (dB) |
| `streams` | Numerical | Historical playback stream volume |
| **`viral` (Target)** | **Binary (0/1)** | **1 = Viral Track, 0 = Non-Viral Track** |

---

## c. GitHub Repository Link
- **GitHub Repository:** [https://github.com/rutaanshnambiar/ML_Assignment_2](https://github.com/rutaanshnambiar/ML_Assignment_2) *(replace with your final repository URL)*
- **Live Streamlit App:** [https://vibe-ai-music-virality.streamlit.app](https://vibe-ai-music-virality.streamlit.app) *(replace with your Streamlit Cloud URL)*

---

## d. Models Used & Comparative Evaluation

All five classifiers were trained on the exact same 8,000-sample training partition and evaluated on the identical 2,000-sample stratified holdout test set (`test_data.csv`). Preprocessing utilized `StandardScaler()` for numerical features and `OneHotEncoder(handle_unknown='ignore')` for categorical features within a `ColumnTransformer` pipeline.

### Master Evaluation Metrics Comparison Table:
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8195** | **0.8966** | **0.8093** | 0.6533 | 0.7229 | 0.5988 |
| **Decision Tree** | 0.8150 | 0.8925 | 0.7388 | 0.7531 | **0.7459** | **0.6005** |
| **KNN** | 0.7190 | 0.7799 | 0.6293 | 0.5368 | 0.5793 | 0.3730 |
| **Naive Bayes** | 0.7495 | 0.8638 | 0.6111 | **0.8391** | 0.7072 | 0.5168 |
| **Support Vector Machine** | 0.7995 | 0.8736 | 0.7941 | 0.5992 | 0.6830 | 0.5520 |

---

### Observations on Model Performance:
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Top Overall Performer for Accuracy (81.95%) and ROC-AUC (0.8966)**. Achieved the highest Precision (80.93%), demonstrating that linear decision boundaries along normalized acoustic features (`danceability`, `energy`, `acousticness`) separate viral from non-viral tracks effectively while minimizing costly false positive predictions. |
| **Decision Tree** | **Top Balanced Model for F1 Score (0.7459) and MCC (0.6005)**. By learning orthogonal splits on non-linear threshold interactions (e.g., high danceability combined with low acousticness), the Decision Tree maintains an optimal equilibrium between Precision (73.88%) and Recall (75.31%). |
| **KNN** | **Lowest Performing Classifier (Accuracy: 71.90%, MCC: 0.3730)**. Suffers from the curse of dimensionality across the 22 transformed feature dimensions (12 standardized numeric + 10 one-hot genre columns), causing local Euclidean neighbor distances to lose discrimination power. |
| **Naive Bayes (Gaussian)** | **Highest Recall Performer (83.91%)**. Excelled at detecting the vast majority of viral tracks, but had lower Precision (61.11%) due to the strong conditional independence assumption among correlated audio attributes (such as energy and loudness). |
| **Support Vector Machine** | **Strong Non-Linear Classifier (Accuracy: 79.95%, Precision: 79.41%)**. The Radial Basis Function (RBF) kernel produced smooth, high-precision decision margins, achieving an AUC of 0.8736. |
| **Overall, Winner for your dataset?** | **Logistic Regression & Decision Tree (Co-Winners)**:<br>• **Logistic Regression** is the premier production model when minimizing false alarms is paramount (highest Accuracy of **81.95%**, highest Precision of **80.93%**, and highest AUC of **0.8966**).<br>• **Decision Tree** is the winner when seeking the most balanced detector of viral trends (highest F1 Score of **0.7459** and highest Matthews Correlation Coefficient of **0.6005**). |

---

## e. Repository Structure
```text
ML_Assignment_2/
├── app.py                      # Interactive Streamlit Web Application (Frontend + Inference)
├── requirements.txt            # Python dependencies for Streamlit Cloud deployment
├── README.md                   # Complete Assignment Report & Documentation
├── test_data.csv               # 2,000-instance holdout test dataset (features + target)
├── train_models.py             # Reproducible training, evaluation & serialization script
├── notebook.ipynb              # Fully executed Jupyter Notebook for BITS Lab screenshot
├── create_notebook.py          # Automated notebook builder & executor
├── Viral_Songs_Datatset/       # Dataset folder containing original CSV & datasheet
│   └── viral_songs.csv         # Full 10,000-row viral songs dataset
└── model/                      # Serialized trained model binaries & preprocessing artifacts
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── preprocessor.joblib     # ColumnTransformer (StandardScaler + OneHotEncoder)
    ├── metadata.joblib         # Feature column names & genre definitions
    └── evaluation_results.json # Cached baseline evaluation metrics
```

---

