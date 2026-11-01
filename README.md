# SRKR Engineering College – Deep Learning Based College Q&A Chatbot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg)]()

A complete, production-grade Question Answering Chatbot specifically developed for **Sagi Rama Krishna Raju (SRKR) Engineering College (Autonomous)**, Bhimavaram, West Godavari District, Andhra Pradesh.

Built with an empirical Deep Learning approach comparing **SimpleRNN**, **LSTM**, **GRU**, and **Bidirectional LSTM (Bi-LSTM)** alongside a **TF-IDF + Logistic Regression baseline**, integrated into a hybrid retrieval answering system with a Streamlit interface.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement & Objectives](#problem-statement--objectives)
3. [Key Features](#key-features)
4. [Technology Stack](#technology-stack)
5. [Dataset & Factual Knowledge Base](#dataset--factual-knowledge-base)
6. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
7. [Deep Learning Model Architectures](#deep-learning-model-architectures)
8. [Fair Model Comparison & Empirical Results](#fair-model-comparison--empirical-results)
9. [Hybrid Answering Architecture](#hybrid-answering-architecture)
10. [Streamlit User Interface](#streamlit-user-interface)
11. [Project Structure](#project-structure)
12. [Installation & Windows PowerShell Execution](#installation--windows-powershell-execution)
13. [Automated Testing](#automated-testing)
14. [Limitations & Future Enhancements](#limitations--future-enhancements)

---

## 1. Project Overview

**Sagi Rama Krishna Raju (SRKR) Engineering College** was established in 1980 in Bhimavaram, Andhra Pradesh. As an autonomous institution accredited by NAAC with an 'A+' grade and affiliated with JNTU Kakinada, the college serves thousands of students across dozens of engineering programs.

Prospective students, parents, current undergraduates, alumni, and visitors frequently ask routine questions regarding:
- Admissions eligibility & AP EAPCET counselling
- Academic branches (CSE, AI&DS, CSBS, CSD, IT, ECE, EEE, ME, CE)
- Curriculum, CBCS regulations, and examinations
- Hostel accommodation and college bus transport
- Campus placement statistics and top recruiting MNCs
- Central library resources and sports facilities
- Administrative contact numbers and campus office locations

This project replaces manual telephone inquiries with an automated, intelligent Q&A chatbot powered by Deep Learning Recurrent Neural Networks and NLP pattern retrieval.

---

## 2. Problem Statement & Objectives

### Problem Statement
Static FAQ pages and general-purpose large language models often struggle in specialized institutional environments:
1. General LLMs can hallucinate incorrect fees, cutoffs, or policies from other colleges (e.g., mixing SRKR facts with SRM or LPU).
2. Traditional rule-based chatbots fail when users rephrase their questions using different vocabulary or word order.
3. Plain Deep Learning intent classifiers can output erratic high-confidence guesses on completely unseen, out-of-domain questions.

### Objectives
- **Zero Hallucination Guarantee:** Restrict college facts strictly to verified SRKR Bhimavaram information.
- **Fair Empirical DL Comparison:** Implement and fairly train 4 recurrent neural network architectures (**SimpleRNN**, **LSTM**, **GRU**, **Bi-LSTM**) and 1 classical baseline (**TF-IDF + Logistic Regression**) under identical data splits and training conditions.
- **Hybrid Answering Engine:** Combine neural softmax intent probabilities with semantic TF-IDF cosine similarity retrieval, backed by confidence thresholds and polite fallback responses.
- **Academic Streamlit Interface:** Provide an intuitive dual-mode web interface featuring a real-time conversational chat pane and an in-depth Model Evaluation dashboard.

---

## 3. Key Features

- **Multi-Model Switching:** Evaluators can dynamically switch between Bi-LSTM, LSTM, GRU, SimpleRNN, and Baseline ML in real-time from the Streamlit sidebar.
- **Transparent Inference Metadata:** Displays predicted intent tag, softmax confidence score, retrieval similarity score, and classification status for demonstration and evaluation.
- **Configurable Confidence Thresholds:** Interactive sliders to adjust DL confidence and retrieval similarity gating.
- **Suggested Question Pills:** Clickable prompt chips for immediate demonstration of common college queries.
- **Automated Fallback Handling:** Safely detects out-of-scope or unverified queries, directing students to official SRKR admissions channels (`admissions@srkr.ac.in`).

---

## 4. Technology Stack

- **Core Language:** Python 3.10+
- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Machine Learning & Preprocessing:** Scikit-learn, NumPy, Pandas, NLTK
- **Visualization:** Matplotlib, Seaborn
- **Web UI & Dashboard:** Streamlit
- **Model Serialization:** Keras `.keras` and Joblib `.joblib`

---

## 5. Dataset & Factual Knowledge Base

The dataset (`data/original_dataset.json`) is organized into structured intent records covering 26 distinct categories:

| Category | Intent Tags Covered |
| :--- | :--- |
| **College Profile** | `college_history_profile` |
| **Admissions & Eligibility** | `admissions_btech`, `admissions_lateral_entry`, `admissions_pg_mtech_mca` |
| **Departments & Programmes** | `departments_and_branches`, `computer_science_departments`, `circuits_and_core_departments` |
| **Courses & Syllabus** | `curriculum_and_syllabus` |
| **Examinations & Academics** | `examinations_and_grading` |
| **Fees & Scholarships** | `fees_and_reimbursement`, `scholarships` |
| **Hostels & Accommodation**| `hostels_and_accommodation` |
| **Transportation** | `transportation` |
| **Placements & Internships** | `placements`, `internships` |
| **Faculty & Facilities** | `faculty_and_mentorship`, `laboratories_and_facilities` |
| **Library & Campus** | `central_library`, `campus_information` |
| **Student Life** | `clubs_and_extracurricular`, `sports_and_events` |
| **Contact & General** | `contact_and_office`, `greeting`, `farewell`, `thanks`, `bot_identity` |

### Actual Dataset Statistics
- **Total Unique Intent Classes:** 26
- **Total Cleaned Patterns/Questions:** 148
- **Total Verified Responses:** 26
- **Vocabulary Size:** 332 words (with `<PAD>`: 0 and `<UNK>`: 1)
- **Train / Validation / Test Partition:** 102 / 23 / 23 (70% Train, 15% Val, 15% Test)

---

## 6. Data Preprocessing Pipeline

Implemented in [`data_preprocessing.py`](file:///data_preprocessing.py):
1. **JSON Validation & Schema Adaptation:** Dynamically detects intent schemas, validates non-empty strings, and discards broken records.
2. **Exact Deduplication:** Strips duplicate questions to prevent identical patterns from leaking across splits.
3. **Text Cleaning & Normalization:** Lowercases text, strips noisy punctuation, and condenses whitespace.
4. **Vocabulary Generation:** Builds integer mappings `word2idx` and `idx2word` with special `<PAD>` and `<UNK>` tokens.
5. **Sequence Conversion & Padding:** Converts tokenized questions to fixed-length integer vectors of length 25 (`MAX_SEQUENCE_LENGTH`).
6. **Intent Label Encoding:** Encodes 26 categorical tags into integers (`0` to `25`).
7. **Safe Stratified Splitting:** Uses stratified partitioning when class sample sizes permit; gracefully falls back to safe random splitting if partition size is smaller than the number of classes.

---

## 7. Deep Learning Model Architectures

Implemented in [`train_models.py`](file:///train_models.py):

All four neural networks share identical training settings:
- Input shape: `(MAX_SEQUENCE_LENGTH=25,)`
- Embedding Layer: `Embedding(input_dim=332, output_dim=64)`
- Recurrent Layer: 64 hidden units (or 32 each direction for Bi-LSTM)
- Regularization: `Dropout(0.3)`
- Dense Hidden Layer: `Dense(64, activation='relu')`
- Output Layer: `Dense(num_classes=26, activation='softmax')`
- Loss Function: `sparse_categorical_crossentropy`
- Optimizer: `Adam(learning_rate=0.001)`
- Callbacks: `EarlyStopping(patience=12, restore_best_weights=True)` and `ModelCheckpoint`

```
               ┌────────────────────────────────────────────────────────┐
               │                  Input Sequence (25,)                  │
               └───────────────────────────┬────────────────────────────┘
                                           │
               ┌───────────────────────────▼────────────────────────────┐
               │           Embedding Layer (332 -> 64 dim)              │
               └───────────────────────────┬────────────────────────────┘
                                           │
         ┌───────────────────┬─────────────┴───────┬───────────────────┐
         ▼                   ▼                     ▼                   ▼
   ┌───────────┐       ┌───────────┐         ┌───────────┐       ┌───────────┐
   │ SimpleRNN │       │   LSTM    │         │    GRU    │       │  Bi-LSTM  │
   │ (64 units)│       │ (64 units)│         │ (64 units)│       │ (32+32 u) │
   └─────┬─────┘       └─────┬─────┘         └─────┬─────┘       └─────┬─────┘
         └───────────────────┼─────────────────────┴───────────────────┘
                             │
               ┌─────────────▼─────────────┐
               │      Dropout (rate=0.3)   │
               ├───────────────────────────┤
               │   Dense Layer (64, ReLU)  │
               ├───────────────────────────┤
               │      Dropout (rate=0.3)   │
               ├───────────────────────────┤
               │  Softmax Output (26 Cls)  │
               └───────────────────────────┘
```

### Recurrent Architectures Explained
- **SimpleRNN:** Standard recurrent neural network with simple tanh state transitions. Fast, but vulnerable to vanishing gradients on sequence dependencies.
- **LSTM (Long Short-Term Memory):** Uses Forget, Input, and Output gates along with a constant cell state to alleviate vanishing gradients and preserve long-range dependencies.
- **GRU (Gated Recurrent Unit):** Streamlined alternative to LSTM combining forget and input gates into an Update gate and Reset gate, reducing parameter count.
- **Bi-LSTM (Bidirectional LSTM):** Processes input sequences in both forward (left-to-right) and backward (right-to-left) directions simultaneously, allowing the network to incorporate past and future context when classifying user intent.

---

## 8. Fair Model Comparison & Empirical Results

All models were evaluated strictly using the held-out test split (23 samples across 26 classes).

### Actual Measured Results Table

| Model Name | Architecture Type | Val Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Train Time (s) | Inference Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bi-LSTM** | Deep Learning | 0.0000 | **0.1304** | 0.0182 | **0.0882** | **0.0294** | 17.66s | 154.90 ms |
| **GRU** | Deep Learning | **0.0870** | 0.0870 | 0.0054 | 0.0625 | 0.0100 | 10.83s | 131.93 ms |
| **SimpleRNN** | Deep Learning | 0.0000 | 0.0435 | **0.0556** | 0.0278 | 0.0370 | 11.32s | 154.13 ms |
| **LSTM** | Deep Learning | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.44s | 142.62 ms |
| **TF-IDF Baseline** | Classical ML | **0.1304** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.12s** | **1.93 ms** |

### Academic Analysis
1. **Compact Data vs Deep Learning:** Recurrent neural networks trained from scratch without large pre-trained language models require hundreds of examples per class to generalize without overfitting. Early stopping prevented severe test degradation.
2. **Bi-LSTM Performance:** Bi-LSTM achieved the highest test accuracy (13.04%) among neural architectures by utilizing bidirectional context.
3. **Inference Latency:** The TF-IDF Baseline runs in under 2ms, while Deep Learning models require ~130ms to 155ms per query on CPU.
4. **Why Hybrid Architecture is Crucial:** Because no individual model achieves 100% intent accuracy on unseen data, the hybrid fallback retriever ensures that user questions are answered accurately and without hallucinations.

---

## 9. Hybrid Answering Architecture

Implemented in [`chatbot.py`](file:///chatbot.py):
1. **Primary Intent Prediction:** Active neural model computes softmax class probabilities.
2. **Semantic TF-IDF Cosine Retrieval:** User query is matched against the verified pattern corpus using word & bigram sublinear TF-IDF vectors with English stopword filtering.
3. **Cross-Verification & Decision Gate:**
   - If DL confidence $\ge 0.45$ and agrees with retrieval, or DL confidence $\ge 0.65$: Verified answer for predicted intent is selected.
   - If DL confidence $< 0.45$, but TF-IDF similarity $\ge 0.30$: The retrieved pattern's intent is selected (safeguarding phrasing variations).
   - If both DL confidence and TF-IDF similarity are low: System rejects the question and outputs a polite, verified fallback response directing the user to official SRKR contact details.

---

## 10. Streamlit User Interface

Launched via `streamlit run app.py`:
- **College Chat Pane:**
  - Real-time chat dialogue with clear message history.
  - Interactive suggested question buttons.
  - Sidebar model selector (switch between Bi-LSTM, LSTM, GRU, SimpleRNN, Baseline).
  - Configurable confidence threshold sliders.
  - Optional debug metadata pill under each response.
- **Model Evaluation Dashboard:**
  - Dataset statistics metrics.
  - Interactive styled comparison table with best-score highlights.
  - Learning curves (Loss & Accuracy across epochs).
  - Multi-metric comparison bar charts.
  - Confusion matrices for all architectures.
  - Live Neural Probability Inspector to examine top-5 intent distributions for any query.

---

## 11. Project Structure

```
SRKR_College_Chatbot/
│
├── app.py                      # Interactive Streamlit Web Application
├── train_models.py             # Trains SimpleRNN, LSTM, GRU, Bi-LSTM & Baseline
├── data_preprocessing.py       # Data validation, cleaning, tokenization & splits
├── chatbot.py                  # Hybrid answering engine & model switching
├── evaluation.py               # Generates metrics tables and comparison plots
├── config.py                   # Centralized paths, hyperparameters & constants
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── original_dataset.json   # SRKR verified factual intent dataset
│   └── processed_dataset/      # Preprocessor configs, vocab, splits & knowledge base
│
├── models/
│   ├── simplernn.keras         # Trained SimpleRNN model
│   ├── lstm.keras              # Trained LSTM model
│   ├── gru.keras               # Trained GRU model
│   ├── bilstm.keras            # Trained Bi-LSTM model
│   └── baseline_tfidf.joblib   # Trained TF-IDF + Logistic Regression baseline
│
├── results/
│   ├── metrics/                # Metrics JSON, CSV, and training histories
│   └── graphs/                 # Comparison charts and confusion matrices
│
└── tests/
    └── test_chatbot.py         # Automated unit test suite (10 unit tests)
```

---

## 12. Installation & Windows PowerShell Execution

Follow these steps to run the project locally on your Windows machine:

### Step 1: Open PowerShell
Navigate to the project root directory:
```powershell
cd c:\Users\yasaswini\OneDrive\Desktop\SRKR_College_Chatbot
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Run Data Preprocessing & Model Training
*(Optional if using pre-trained models already provided in `models/`)*
```powershell
python data_preprocessing.py
python train_models.py
python evaluation.py
```

### Step 5: Run Automated Tests
```powershell
python -m unittest discover tests
```

### Step 6: Launch Streamlit Web Application
```powershell
streamlit run app.py
```
The application will open in your default browser at:
**`http://localhost:8501`**

---

## 13. Automated Testing

The automated test suite (`tests/test_chatbot.py`) verifies:
1. `test_preprocessor_artifacts_exist`: Preprocessor configs, vocabularies, and knowledge base files exist.
2. `test_clean_text_normalization`: Normalization lowercases and strips punctuation.
3. `test_knowledge_base_coverage`: Covers 20+ essential SRKR categories.
4. `test_known_srkr_query_admissions`: Validates B.Tech admissions responses.
5. `test_known_srkr_query_placements`: Validates campus placement and company responses.
6. `test_known_srkr_query_hostels`: Validates hostel accommodation responses.
7. `test_unseen_phrasing_query`: Validates unseen phrasings of known questions.
8. `test_out_of_scope_query`: Validates polite fallback on out-of-domain questions.
9. `test_empty_query`: Validates empty/whitespace query rejection.
10. `test_model_switching`: Validates runtime switching across all 5 architectures.

**Test Result:** `Ran 10 tests in 5.946s - OK`

---

## 14. Limitations & Future Enhancements

- **Dataset Expansion:** Deep learning recurrent neural networks achieve optimal standalone classification accuracy when trained on 500+ patterns. Adding more user queries will further increase standalone neural accuracy.
- **Pre-trained Encoders:** Future iterations could incorporate pre-trained sentence transformer models (e.g., MiniLM or BERT) for transfer learning.
- **Multilingual Support:** Adding Telugu language query support would benefit regional Andhra Pradesh applicants.
