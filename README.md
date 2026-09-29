# AI Text Detection using Stylometric Features and Deep Learning

An advanced Machine Learning and Natural Language Processing (NLP) framework designed to detect AI-generated text. This project explores and compares two distinct paradigms in data science: **Classical Machine Learning with Stylometric Feature Engineering** versus **Modern Deep Learning using Transformer Architectures**.

## 🚀 Project Overview

The rise of Large Language Models (LLMs) has made distinguishing between human-written and machine-generated text a critical challenge. This repository implements an end-to-end pipeline that addresses this challenge through a rigorous data science workflow: from dataset balancing and custom linguistic feature engineering to dimensionality reduction and advanced neural network fine-tuning.

## 📊 Dataset & Preprocessing

* **Data Source:** Fetched directly from the `artem9k/ai-text-detection-pile` dataset on Hugging Face.
* **Class Imbalance Mitigation:** The raw dataset contains a severe class imbalance. To ensure robust training, a proportional down-sampling workflow was implemented, resulting in a perfectly balanced subset of **70,000 text samples** (35,000 Human / 35,000 AI).
* **Data Splits:** Scaled and split into Training (70%), Validation (15%), and Testing (15%) subsets using stratified splitting to maintain identical class distributions across all sets.

## 🛠️ Feature Engineering & Dimensionality Reduction

One of the core strengths of this project is the construction of a custom stylometric feature extractor. Instead of relying solely on token frequencies, the pipeline extracts **34 distinct linguistic, punctuation, and stylometric features** to capture writing style fingerprints:

1. **Lexical Features:** Character count, word count, average word length, Type-Token Ratio (vocabulary richness), short-word ratio, long-word ratio, and average sentence length.
2. **Punctuation Features:** Frequency of periods, commas, question marks, exclamation marks, semicolons, colons, dashes, quotes, apostrophes, parentheses, and ellipses.
3. **Stylometric Features:** Uppercase character ratio, digit ratio, whitespace ratio, paragraph count, average words per paragraph, repeated word ratio, capital word ratio, all-caps word ratio, contraction frequency, and sentence length standard deviation.

### Dimensionality Reduction (PCA)
Before modeling, **Principal Component Analysis (PCA)** was applied to the standardized 34-dimensional feature space. Reducing the data to 2 principal components (PC1 and PC2) visually demonstrated a clear mathematical separation (clustering) between human and AI-generated text styles, proving the strong predictive power of the engineered features.

## 🤖 Models & Architecture

The framework evaluates and compares two completely different architectures:

### 1. Random Forest Classifier (Classical ML)
* Trained on the 34 custom engineered features.
* **Hyperparameters:** `n_estimators=100`, `max_depth=15`, `min_samples_split=5`.
* **Feature Importance:** Explracted the internal Gini importance scores, revealing that **average sentence length** (`avg_chars_per_sentence`) and **lexical diversity** (`type_token_ratio`) are the single most discriminative markers of AI authorship.

### 2. DistilBERT (Deep Learning / Transformer)
* Fine-tuned a pre-trained `distilbert-base-uncased` transformer model for sequence classification.
* **Training Dynamics:** Trained for 3 epochs using the AdamW optimizer, a linear learning rate scheduler with warmup, gradient norm clipping (`max_norm=1.0`), and **Label Smoothing (0.05)** to prevent overfitting and improve generalization.

## 🏆 Model Comparison & Evaluation

The experimental results demonstrate a classic engineering trade-off between predictive maximums and computational efficiency:

| Metric | Random Forest (Classical ML) | DistilBERT (Transformer) | Winner |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 89.24% | **95.77%** | **Transformer** |
| **Precision** | 0.8507 | **0.9294** | **Transformer** |
| **Recall** | 0.9518 | **0.9907** | **Transformer** |
| **F1 Score** | 0.8984 | **0.9591** | **Transformer** |
| **ROC-AUC** | 0.9580 | **0.9859** | **Transformer** |
| **Train Time** | **21.5 sec** | ~20 min (GPU) | **RF** |


### Key Takeaways
* **Transformer Superiority:** DistilBERT achieves near-perfect classification scores by deeply capturing contextual and semantic boundaries that hardcoded features miss.
* **Feature Engineering Efficiency:** The Random Forest model provides an exceptional performance-to-compute ratio, offering high-tier accuracy in a fraction of the training time, making it ideal for resource-constrained or real-time deployment environments.

## 📂 Repository Structure

```text
├── split_train.parquet          # Tokenized training split
├── split_val.parquet            # Tokenized validation split
├── split_test.parquet           # Tokenized test split
├── data1.csv                    # Extracted linguistic features dataset
├── rf_misclassified.csv         # Random Forest error analysis logs
├── transformer_misclassified.csv # DistilBERT error analysis logs
├── rf_confusion_matrix.png     # RF Evaluation plot
├── rf_feature_importance.png    # RF Feature ranking plot
├── rf_roc_curves.png            # RF ROC Curve plot
├── pca_2d_visualization.png     # 2D PCA Cluster visualization
├── model_comparison_summary.csv # Final side-by-side metrics
└── outputs/
    ├── best_model.pt            # Best weights for DistilBERT
    ├── training_curves.png      # Loss/Accuracy plots over epochs
    └── tokenizer/               # Saved model tokenizer
```

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com
cd AI-Text-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Make sure to have `torch`, `transformers`, `scikit-learn`, `pandas`, `pyarrow`, `matplotlib`, and `seaborn` installed).*

### 3. Running the Pipeline
You can run the full exploratory data analysis, feature extraction, and modeling script using python or Jupyter Notebook.
```bash
# To run the core architecture and reproduce results
python main.py
```
