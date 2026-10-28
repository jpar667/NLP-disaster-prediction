"""Frozen pretrained sentence embeddings with a source-validated local cache."""
from pathlib import Path
import hashlib
import html
import json
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from disaster_tweets.model_pipeline import make_model
from disaster_tweets.preprocessing import read_training_csv

from disaster_tweets.paths import PROJECT_ROOT, RAW_DATA_DIR, EMBEDDING_CACHE_DIR

ROOT = PROJECT_ROOT
CACHE = EMBEDDING_CACHE_DIR
MODEL_ID = 'sentence-transformers/all-mpnet-base-v2'
MODEL_REVISION = 'e8c3b32edf5434bc2275fc9bab85f82640a19130'
PREPROCESSING_VERSION = 1
_LOOKUP = None


def embedding_text(value):
    value = html.unescape(value)
    value = re.sub(r'https?://\S+|www\.\S+', ' ', value, flags=re.I)
    value = re.sub(r'(?<!\w)@\w+', ' @user ', value)
    return re.sub(r'\s+', ' ', value).strip()


def prepare_embeddings():
    global _LOOKUP
    CACHE.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: hashlib.sha256((RAW_DATA_DIR / name).read_bytes()).hexdigest()
                     for name in ['train.csv', 'test.csv']}
    matrix_path = CACHE / 'embeddings.npz'
    metadata_path = CACHE / 'embedding_manifest.json'
    if metadata_path.exists() and matrix_path.exists():
        manifest = json.loads(metadata_path.read_text())
        if (manifest['source_sha256'] == source_hashes and manifest['model_id'] == MODEL_ID
                and manifest['revision'] == MODEL_REVISION
                and manifest['preprocessing_version'] == PREPROCESSING_VERSION
                and manifest['cache_sha256'] == hashlib.sha256(matrix_path.read_bytes()).hexdigest()):
            with np.load(matrix_path, allow_pickle=False) as cached:
                texts, vectors = cached['texts'], cached['vectors']
            _LOOKUP = {str(t): v for t, v in zip(texts, vectors)}
            print(f'Loaded {len(texts):,} frozen sentence embeddings from cache.', flush=True)
            return manifest

    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    revision = MODEL_REVISION
    print(f'Loading {MODEL_ID} revision {revision} on {device}...', flush=True)
    encoder = SentenceTransformer(MODEL_ID, revision=revision, device=device, trust_remote_code=False)
    encoder.max_seq_length = 128
    train = read_training_csv(RAW_DATA_DIR / 'train.csv')
    test = read_training_csv(RAW_DATA_DIR / 'test.csv')
    texts = np.asarray(pd.concat([train['text'], test['text']], ignore_index=True).drop_duplicates().tolist())
    prepared = [embedding_text(t) for t in texts]
    token_lengths = [len(ids) for ids in encoder.tokenizer(prepared, truncation=False)['input_ids']]
    chunks = []
    started = time.perf_counter()
    for start in range(0, len(prepared), 256):
        chunk = encoder.encode(prepared[start:start+256], batch_size=32, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)
        chunks.append(chunk.astype(np.float32))
        print(f'Encoded {min(start+256, len(prepared)):,}/{len(prepared):,} unique tweets '
              f'({time.perf_counter()-started:.1f}s)', flush=True)
    vectors = np.concatenate(chunks)
    assert vectors.shape == (len(texts), 768) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    np.savez_compressed(matrix_path, texts=texts, vectors=vectors)
    manifest = {'model_id': MODEL_ID, 'revision': revision, 'device': device,
                 'preprocessing_version': PREPROCESSING_VERSION, 'source_sha256': source_hashes,
                 'cache_sha256': hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                 'embedding_dimension': 768, 'unique_text_count': len(texts),
                 'max_seq_length': 128, 'max_observed_tokens': max(token_lengths),
                 'truncated_text_count': sum(n > 128 for n in token_lengths),
                 'l2_normalized': True, 'frozen_encoder': True,
                 'target_labels_used_for_embeddings': False,
                 'encoding_seconds': time.perf_counter()-started}
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    _LOOKUP = {str(t): v for t, v in zip(texts, vectors)}
    return manifest


def feature_pairs(fit_frame, evaluation_frame):
    """Only the fixed encoder is shared; learned text/category features fit locally."""
    if _LOOKUP is None:
        prepare_embeddings()
    fit_embedding = np.asarray([_LOOKUP[t] for t in fit_frame['text']], dtype=np.float64)
    eval_embedding = np.asarray([_LOOKUP[t] for t in evaluation_frame['text']], dtype=np.float64)
    keyword = OneHotEncoder(handle_unknown='ignore')
    key_fit = .5 * keyword.fit_transform(fit_frame[['keyword']])
    key_eval = .5 * keyword.transform(evaluation_frame[['keyword']])
    original = make_model(['keyword']).named_steps['features']
    original_fit = original.fit_transform(fit_frame)
    original_eval = original.transform(evaluation_frame)
    return {
        'original': (original_fit, original_eval),
        'embedding': (fit_embedding, eval_embedding),
        'embedding_keyword': (sparse.hstack([fit_embedding, key_fit], format='csr'),
                               sparse.hstack([eval_embedding, key_eval], format='csr')),
        'hybrid': (sparse.hstack([original_fit, 2*fit_embedding], format='csr'),
                    sparse.hstack([original_eval, 2*eval_embedding], format='csr')),
    }


if __name__ == '__main__':
    prepare_embeddings()
