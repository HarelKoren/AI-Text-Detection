# AI Text Detection Project

An advanced Machine Learning and NLP project designed to classify and detect AI-generated text from human-written text.

##  Models & Architecture
The project evaluates and compares multiple machine learning algorithms, ranging from traditional classifiers to state-of-the-art deep learning architectures:

* **Traditional Classifiers:**
  * **Logistic Regression** – Used as a fast and interpretable baseline model.
  * **Multinomial Naive Bayes** – A probabilistic classifier highly effective for text data.
  * **Random Forest (RF)** – An ensemble learning method utilizing multiple decision trees for robust classification.

* **Advanced Deep Learning:**
  * **Transformers** – Utilizing state-of-the-art Deep Learning models for Natural Language Processing (NLP) to capture context and semantic meaning within the text.

## 📊 Dataset & Pipeline
* **Data Source:** Fetched directly from the Hugging Face ecosystem (`artem9k/ai-text-detection-pile`).
* **Feature Extraction:** Standard text data is vectorized using **TF-IDF** for traditional models, while raw text embeddings are used for the Transformer model.

## 🛠️ Technologies Used
* **Python**
* **Scikit-Learn** (Logistic Regression, Naive Bayes, Random Forest)
* **Hugging Face / Transformers** (Deep Learning NLP models)
* **Pandas & NumPy** (Data Preprocessing)
* **Jupyter Notebook / Google Colab**
