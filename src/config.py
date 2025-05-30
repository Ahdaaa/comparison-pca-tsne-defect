# ✅ NEW: relative to project root no matter where you're running from
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # go up from src/
ARFF_FOLDER_PATH = os.path.join(PROJECT_ROOT, "data")
