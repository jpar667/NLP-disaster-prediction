"""Deterministic metadata cleaning; raw text and labels are preserved."""
from pathlib import Path
import html
import re
import unicodedata

import pandas as pd

MISSING_TOKEN = '__MISSING__'
METADATA_COLUMNS = ('keyword', 'location')


def read_training_csv(path: Path) -> pd.DataFrame:
    # Preserve literal strings such as "NA"; only empty/whitespace metadata is missing.
    return pd.read_csv(path, keep_default_na=False,
                       dtype={'keyword': 'string', 'location': 'string', 'text': 'string'})


def clean_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with trimmed metadata and explicit missing-value markers.

    No rows are dropped, and text, ID, and target are left unchanged. This
    transformation learns nothing from the labels or from validation rows.
    """
    result = frame.copy()
    for column in METADATA_COLUMNS:
        values = result[column].astype('string').str.strip()
        result[column] = values.replace('', pd.NA).fillna(MISSING_TOKEN)
    return result


def normalize_for_grouping(value: str) -> str:
    """Match the conservative duplicate rule used by the EDA."""
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', html.unescape(value)).casefold()).strip()
