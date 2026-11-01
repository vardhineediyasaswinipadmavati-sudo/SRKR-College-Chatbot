import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Check for existing SRKR JSON dataset
user_dataset = BASE_DIR / "SRKR_detailed_chatbot_dataset_v2.json"
if user_dataset.exists():
    ORIGINAL_DATASET_PATH = user_dataset
elif (DATA_DIR / "original_dataset.json").exists():
    ORIGINAL_DATASET_PATH = DATA_DIR / "original_dataset.json"
else:
    ORIGINAL_DATASET_PATH = user_dataset
PROCESSED_DATA_DIR = DATA_DIR / "processed_dataset"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
GRAPHS_DIR = RESULTS_DIR / "graphs"

# Ensure runtime directories exist
for path in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, METRICS_DIR, GRAPHS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Preprocessing Hyperparameters
MAX_VOCAB_SIZE = 1500
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 64
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
RANDOM_STATE = 42

# Training Hyperparameters
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
RECURRENT_UNITS = 64
DENSE_UNITS = 64
EARLY_STOPPING_PATIENCE = 12

# Chatbot Answering & Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.45
SIMILARITY_THRESHOLD = 0.30

# SRKR Official Information & Fallback Responses
COLLEGE_NAME = "Sagi Rama Krishnam Raju (SRKR) Engineering College"
COLLEGE_LOCATION = "SRKR Marg, China Amiram, Bhimavaram, West Godavari District, Andhra Pradesh, India - 534204"
COLLEGE_WEBSITE = "https://www.srkrec.ac.in"
COLLEGE_PHONE = "+91 9848823332"
COLLEGE_EMAIL = "info@srkrec.ac.in"

DEFAULT_FALLBACK_RESPONSE = (
    "I am trained specifically on verified facts about Sagi Rama Krishnam Raju (SRKR) Engineering College, Bhimavaram. "
    "I do not have verified information answering your question in my knowledge base. "
    "For official confirmation, please visit the official college website at https://www.srkrec.ac.in, "
    "or contact the college office at +91 9848823332 / email info@srkrec.ac.in."
)
