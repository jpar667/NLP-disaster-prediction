"""Project paths independent of the current working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
REPORTS_DIR = PROJECT_ROOT / 'reports'
EDA_REPORTS_DIR = REPORTS_DIR / 'eda'
BASELINE_REPORTS_DIR = REPORTS_DIR / 'baseline'
SPARSE_REPORTS_DIR = REPORTS_DIR / 'sparse'
SEMANTIC_REPORTS_DIR = REPORTS_DIR / 'semantic'
SUBMISSION_REPORTS_DIR = REPORTS_DIR / 'submission'
SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
HTML_REPORTS_DIR = REPORTS_DIR / 'html'
EMBEDDING_CACHE_DIR = PROJECT_ROOT / '.cache' / 'sentence_embeddings'
