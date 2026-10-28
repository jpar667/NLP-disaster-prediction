"""Fit the agreed text + keyword baseline on all train rows and predict test.csv."""
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from disaster_tweets.model_pipeline import make_model
from disaster_tweets.preprocessing import clean_metadata, read_training_csv

from disaster_tweets.paths import PROJECT_ROOT, RAW_DATA_DIR, SUBMISSIONS_DIR, BASELINE_REPORTS_DIR, SUBMISSION_REPORTS_DIR

ROOT = PROJECT_ROOT


def main():
    source_hashes = {name: hashlib.sha256((RAW_DATA_DIR / name).read_bytes()).hexdigest()
                     for name in ['train.csv', 'test.csv', 'sample_submission.csv']}
    train = clean_metadata(read_training_csv(RAW_DATA_DIR / 'train.csv'))
    test = clean_metadata(read_training_csv(RAW_DATA_DIR / 'test.csv'))
    template = pd.read_csv(RAW_DATA_DIR / 'sample_submission.csv')
    assert train['id'].is_unique and test['id'].is_unique and template['id'].is_unique
    assert len(template) == len(test) and set(template['id']) == set(test['id'])
    assert not set(train['id']).intersection(test['id'])
    assert list(template.columns) == ['id', 'target']
    assert train['target'].isin([0, 1]).all() and set(train['target']) == {0, 1}
    for frame in [train, test]:
        assert frame[['text', 'keyword']].notna().all().all()
        assert frame['text'].str.strip().ne('').all()

    # Reuse the exact feature extraction and classifier settings from validation.
    # Vocabulary, IDF and keyword categories are fitted on train.csv only.
    model = make_model(['keyword'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(train[['text', 'keyword']], train['target'])
    positive_column = model.classes_.tolist().index(1)
    probability = model.predict_proba(test[['text', 'keyword']])[:, positive_column]
    assert np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all()
    probabilities_by_id = pd.Series(probability, index=test['id'])
    ordered_probability = template['id'].map(probabilities_by_id)
    assert ordered_probability.notna().all()
    submission = template.copy()
    submission['target'] = (ordered_probability >= 0.5).astype('int64')
    submission.to_csv(SUBMISSIONS_DIR / 'submission.csv', index=False)

    # Re-read the actual file to verify the upload artifact, including ID order.
    saved = pd.read_csv(SUBMISSIONS_DIR / 'submission.csv')
    pd.testing.assert_frame_equal(saved, submission)
    assert saved['id'].tolist() == template['id'].tolist()
    assert saved['target'].isin([0, 1]).all() and saved.notna().all().all()

    output = SUBMISSION_REPORTS_DIR
    output.mkdir(exist_ok=True)
    pd.DataFrame({'id': template['id'], 'probability_disaster': ordered_probability}).to_csv(
        output / 'test_probabilities.csv', index=False)
    validation_path = BASELINE_REPORTS_DIR / 'model_comparison.csv'
    validation_reference = None
    if validation_path.exists():
        validation_metrics_path = BASELINE_REPORTS_DIR / 'metrics.json'
        if validation_metrics_path.exists():
            validation_hash = json.loads(validation_metrics_path.read_text())['source_sha256']
            if validation_hash == source_hashes['train.csv']:
                results = pd.read_csv(validation_path)
                result = results.loc[results['model'].eq('text_keyword')].iloc[0]
                validation_reference = {'mean_fold_f1': float(result['mean_fold_f1']),
                                        'oof_f1': float(result['oof_f1']),
                                        'evaluation': 'five stratified normalized-text-grouped folds; not a test score'}
    metadata = {
        'model': 'text_keyword', 'input_columns': ['text', 'keyword'],
        'pipeline_definition': 'model_pipeline.make_model(["keyword"])',
        'training_rows': len(train), 'prediction_rows': len(test), 'threshold': 0.5,
        'source_sha256': source_hashes,
        'submission_sha256': hashlib.sha256((SUBMISSIONS_DIR / 'submission.csv').read_bytes()).hexdigest(),
        'prediction_counts': {str(label): int(saved['target'].eq(label).sum()) for label in [0, 1]},
        'validation_reference': validation_reference,
        'submitted_to_kaggle': False,
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for name, expected in source_hashes.items():
        assert hashlib.sha256((RAW_DATA_DIR / name).read_bytes()).hexdigest() == expected
    print(f'Saved submission.csv: {len(saved):,} rows, columns {list(saved.columns)}')
    print(f'Prediction counts: {metadata["prediction_counts"]}')
    print('Verified sample ID order, binary integer labels, no missing values, and unchanged input files.')


if __name__ == '__main__':
    main()
