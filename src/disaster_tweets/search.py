"""Compare a predefined sparse-text search with the baseline using nested grouped CV.

Only the supplied train.csv labels are used. Existing folds, source CSVs, and
submission.csv are preserved. Final predictions go to submission_sparse_candidate.csv.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import time
import unicodedata
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from disaster_tweets.model_pipeline import make_model
from disaster_tweets.preprocessing import clean_metadata, normalize_for_grouping, read_training_csv

from disaster_tweets.paths import PROJECT_ROOT, RAW_DATA_DIR, BASELINE_REPORTS_DIR, SPARSE_REPORTS_DIR, SUBMISSIONS_DIR

ROOT = PROJECT_ROOT
OUT = SPARSE_REPORTS_DIR
SEED = 42
# Thresholds are on decision_function margins. Zero equals probability 0.5 for LR.
THRESHOLDS = np.round(np.arange(-0.6, 0.41, 0.1), 8)
SPECS = [
    {'name': 'original_lr', 'features': 'original', 'classifier': 'lr', 'C': 1.0},
    {'name': 'clean_word_lr_1', 'features': 'word', 'classifier': 'lr', 'C': 1.0},
    {'name': 'clean_word_lr_4', 'features': 'word', 'classifier': 'lr', 'C': 4.0},
    {'name': 'clean_wordchar_keyword_lr_1', 'features': 'wordchar_keyword', 'classifier': 'lr', 'C': 1.0},
    {'name': 'clean_wordchar_keyword_lr_4', 'features': 'wordchar_keyword', 'classifier': 'lr', 'C': 4.0},
    {'name': 'clean_word_svc_05', 'features': 'word', 'classifier': 'svc', 'C': 0.5},
    {'name': 'clean_word_svc_1', 'features': 'word', 'classifier': 'svc', 'C': 1.0},
    {'name': 'clean_wordchar_svc_05', 'features': 'wordchar', 'classifier': 'svc', 'C': 0.5},
    {'name': 'clean_wordchar_svc_1', 'features': 'wordchar', 'classifier': 'svc', 'C': 1.0},
    {'name': 'nb_word_lr_1', 'features': 'binary_word', 'classifier': 'nb_lr', 'C': 1.0},
    {'name': 'nb_word_lr_4', 'features': 'binary_word', 'classifier': 'nb_lr', 'C': 4.0},
    {'name': 'nb_word_svc_05', 'features': 'binary_word', 'classifier': 'nb_svc', 'C': 0.5},
    {'name': 'nb_word_svc_1', 'features': 'binary_word', 'classifier': 'nb_svc', 'C': 1.0},
]
BY_NAME = {s['name']: s for s in SPECS}


def clean_tweet(value):
    value = unicodedata.normalize('NFKC', html.unescape(value)).replace('’', "'").lower()
    value = re.sub(r'https?://\S+|www\.\S+', ' urltoken ', value)
    value = re.sub(r'(?<!\w)@\w+', ' usertoken ', value)
    value = value.replace("can't", 'can not').replace("won't", 'will not')
    value = re.sub(r"n['’]t\b", ' not', value)
    return re.sub(r'\s+', ' ', value).strip()


def feature_pairs(fit_frame, evaluation_frame):
    """Fit all feature dictionaries on fit_frame only and reuse per split."""
    fit_text = fit_frame['text'].map(clean_tweet)
    evaluation_text = evaluation_frame['text'].map(clean_tweet)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.98,
                           max_features=50_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2,
                           max_features=60_000, sublinear_tf=True)
    binary = CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=.98,
                              max_features=100_000, binary=True,
                              token_pattern=r'(?u)\b\w+\b', dtype=np.float64)
    word_fit = word.fit_transform(fit_text)
    word_eval = word.transform(evaluation_text)
    char_fit = .5 * char.fit_transform(fit_text)
    char_eval = .5 * char.transform(evaluation_text)
    wc_fit = sparse.hstack([word_fit, char_fit], format='csr')
    wc_eval = sparse.hstack([word_eval, char_eval], format='csr')
    keyword = OneHotEncoder(handle_unknown='ignore')
    key_fit = keyword.fit_transform(fit_frame[['keyword']])
    key_eval = keyword.transform(evaluation_frame[['keyword']])
    original = make_model(['keyword']).named_steps['features']
    return {
        'original': (original.fit_transform(fit_frame), original.transform(evaluation_frame)),
        'word': (word_fit, word_eval),
        'wordchar': (wc_fit, wc_eval),
        'wordchar_keyword': (sparse.hstack([wc_fit, key_fit], format='csr'),
                             sparse.hstack([wc_eval, key_eval], format='csr')),
        'binary_word': (binary.fit_transform(fit_text), binary.transform(evaluation_text)),
    }


def fit_decision_scores(spec, fit_matrix, fit_y, evaluation_matrix):
    if spec['classifier'].startswith('nb_'):
        # Supervised feature weights are fitted from this training subset only.
        positive = 1 + np.asarray(fit_matrix[fit_y == 1].sum(axis=0)).ravel()
        negative = 1 + np.asarray(fit_matrix[fit_y == 0].sum(axis=0)).ravel()
        ratio = np.log((positive / positive.sum()) / (negative / negative.sum()))
        fit_matrix = fit_matrix.multiply(ratio).tocsr()
        evaluation_matrix = evaluation_matrix.multiply(ratio).tocsr()
    if spec['classifier'].endswith('svc'):
        classifier = LinearSVC(C=spec['C'], random_state=SEED, max_iter=10_000)
    else:
        classifier = LogisticRegression(C=spec['C'], solver='liblinear',
                                          random_state=SEED, max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        classifier.fit(fit_matrix, fit_y)
    result = classifier.decision_function(evaluation_matrix)
    assert np.isfinite(result).all()
    return result


def score_all(fit_frame, fit_y, evaluation_frame, *, specs=SPECS, feature_builder=feature_pairs):
    pairs = feature_builder(fit_frame, evaluation_frame)
    return {spec['name']: fit_decision_scores(spec, pairs[spec['features']][0], fit_y,
                                              pairs[spec['features']][1]) for spec in specs}


def measures(truth, predictions):
    return {'f1': float(f1_score(truth, predictions)),
            'precision': float(precision_score(truth, predictions, zero_division=0)),
            'recall': float(recall_score(truth, predictions, zero_division=0)),
            'accuracy': float(accuracy_score(truth, predictions))}


def best_threshold(truth, margins):
    predicted = margins[:, None] >= THRESHOLDS
    positive = truth[:, None] == 1
    tp = (predicted & positive).sum(axis=0)
    fp = (predicted & ~positive).sum(axis=0)
    fn = (~predicted & positive).sum(axis=0)
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    # In ties prefer the threshold nearest the untuned default, then the lower one.
    index = min(range(len(THRESHOLDS)), key=lambda i: (-f1[i], abs(THRESHOLDS[i]), THRESHOLDS[i]))
    return float(THRESHOLDS[index]), float(f1[index])


def rank_candidates(truth, margins, specs=SPECS):
    rows = []
    for spec in specs:
        threshold, score = best_threshold(truth, margins[spec['name']])
        rows.append({'model': spec['name'], 'threshold_margin': threshold,
                      'tuning_f1': score, 'default_f1': float(f1_score(truth, margins[spec['name']] >= 0))})
    return pd.DataFrame(rows).sort_values(['tuning_f1', 'model'], ascending=[False, True]).reset_index(drop=True)


def md_table(frame):
    body = [[f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in row]
             for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                       '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                      ['| ' + ' | '.join(row) + ' |' for row in body])


def main(*, feature_builder=feature_pairs, specs=SPECS, out_dir=OUT,
         submission_name='submission_sparse_candidate.csv', method_description=None,
         extra_metadata=None, reproduce_command='python scripts/improve_model.py'):
    OUT = out_dir
    SPECS = specs
    BY_NAME = {s['name']: s for s in SPECS}
    if method_description is None:
        method_description = '''New representations unescape HTML, normalize Unicode NFKC and curly apostrophes, lowercase, replace URLs/@mentions with tokens, expand can't/won't/n't, and collapse whitespace. Raw CSV text remains unchanged. Stopwords, hashtag words, and negation are retained.

- Clean word TF-IDF: 1–2 grams, min_df=2, max_df=.98, max_features=50,000, sublinear TF, L2 normalization.
- Clean character TF-IDF: char_wb 3–5 grams, min_df=2, max_features=60,000, sublinear TF, L2 normalization; character block weight .5. Optional keyword one-hot block weight 1.
- Binary word counts: 1–2 grams, min_df=2, max_df=.98, max_features=100,000, token length at least one. For NB-weighted models, features are multiplied by the training-only smoothed class log-count ratio (alpha=1) before fitting a linear classifier. This is inspired by [Wang and Manning (2012)](https://aclanthology.org/P12-2018/); it is not a claim to reproduce every setting in that paper.
- Logistic regression: liblinear, C in the candidate table, max_iter=1000. LinearSVC: L2 squared-hinge loss, C in the candidate table, max_iter=10,000. Random state 42, no class weighting. Locations and IDs are excluded. See `metrics.json` for every candidate.'''
    OUT.mkdir(exist_ok=True)
    input_names = ['data/raw/train.csv', 'data/raw/test.csv', 'data/raw/sample_submission.csv', 'submissions/submission.csv',
                    'reports/baseline/fold_assignments.csv', 'reports/baseline/oof_predictions.csv']
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in input_names}
    started = time.perf_counter()
    train = clean_metadata(read_training_csv(RAW_DATA_DIR / 'train.csv'))
    X = train[['text', 'keyword']]
    y = train['target'].to_numpy()
    groups = train['text'].map(normalize_for_grouping).to_numpy()
    fold_assignment = pd.read_csv(BASELINE_REPORTS_DIR / 'fold_assignments.csv')
    baseline_oof = pd.read_csv(BASELINE_REPORTS_DIR / 'oof_predictions.csv')
    assert fold_assignment.id.tolist() == train.id.tolist() == baseline_oof.id.tolist()
    assert np.array_equal(baseline_oof.target, y)
    fold_ids = fold_assignment.fold.to_numpy()
    assert pd.DataFrame({'group': groups, 'fold': fold_ids}).groupby('group').fold.nunique().max() == 1
    nested_prediction = np.full(len(y), -1, dtype=int)
    nested_margins = np.full(len(y), np.nan)
    candidate_margins = {spec['name']: np.full(len(y), np.nan) for spec in SPECS}
    nested_rows, inner_rankings, candidate_fold_rows = [], [], []

    for fold in range(1, 6):
        fit_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        assert not set(groups[fit_idx]) & set(groups[valid_idx])
        inner_scores = {spec['name']: np.full(len(fit_idx), np.nan) for spec in SPECS}
        inner_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + fold)
        print(f'Outer fold {fold}/5: tuning on three grouped inner folds...', flush=True)
        for inner_fold, (inner_fit, inner_valid) in enumerate(
                inner_splitter.split(X.iloc[fit_idx], y[fit_idx], groups[fit_idx]), start=1):
            fit_global, valid_global = fit_idx[inner_fit], fit_idx[inner_valid]
            assert not set(groups[fit_global]) & set(groups[valid_global])
            scores = score_all(X.iloc[fit_global], y[fit_global], X.iloc[valid_global],
                               specs=SPECS, feature_builder=feature_builder)
            for name, margin in scores.items():
                inner_scores[name][inner_valid] = margin
            print(f'  Inner fold {inner_fold}/3 complete', flush=True)
        assert all(np.isfinite(v).all() for v in inner_scores.values())
        ranking = rank_candidates(y[fit_idx], inner_scores, specs=SPECS)
        selected = ranking.iloc[0]
        inner_rankings.append(ranking.assign(outer_fold=fold))
        # Outer labels have not been used to select this fold's model or threshold.
        outer_scores = score_all(X.iloc[fit_idx], y[fit_idx], X.iloc[valid_idx],
                                 specs=SPECS, feature_builder=feature_builder)
        selected_margin = outer_scores[selected.model]
        prediction = (selected_margin >= selected.threshold_margin).astype(int)
        nested_prediction[valid_idx] = prediction
        nested_margins[valid_idx] = selected_margin
        row = {'fold': fold, 'selected_model': selected.model,
                'threshold_margin': float(selected.threshold_margin), 'inner_tuning_f1': float(selected.tuning_f1),
                'validation_rows': len(valid_idx), **measures(y[valid_idx], prediction),
                'baseline_f1': float(f1_score(y[valid_idx], baseline_oof.loc[valid_idx, 'text_keyword_prediction']))}
        row['delta_f1_vs_baseline'] = row['f1'] - row['baseline_f1']
        nested_rows.append(row)
        for name, margin in outer_scores.items():
            candidate_margins[name][valid_idx] = margin
            candidate_fold_rows.append({'fold': fold, 'model': name, **measures(y[valid_idx], margin >= 0)})
        pd.DataFrame(nested_rows).to_csv(OUT / 'nested_fold_metrics.csv', index=False)
        print(f'Outer fold {fold}: {selected.model}, margin threshold={selected.threshold_margin:.2f}, '
              f'F1={row["f1"]:.4f}, baseline={row["baseline_f1"]:.4f}', flush=True)

    assert (nested_prediction >= 0).all() and np.isfinite(nested_margins).all()
    assert all(np.isfinite(v).all() for v in candidate_margins.values())
    # Reproducing the original pipeline verifies that the comparison uses identical folds/settings.
    assert np.array_equal(candidate_margins['original_lr'] >= 0, baseline_oof.text_keyword_prediction)
    original_probability = 1 / (1 + np.exp(-candidate_margins['original_lr']))
    assert np.allclose(original_probability, baseline_oof.text_keyword_probability, atol=1e-12)
    nested_scores = measures(y, nested_prediction)
    baseline_scores = measures(y, baseline_oof.text_keyword_prediction)
    nested_frame = pd.DataFrame(nested_rows)
    nested_frame.to_csv(OUT / 'nested_fold_metrics.csv', index=False)
    pd.concat(inner_rankings, ignore_index=True).to_csv(OUT / 'inner_search.csv', index=False)
    pd.DataFrame(candidate_fold_rows).to_csv(OUT / 'candidate_fold_metrics.csv', index=False)
    ranking = rank_candidates(y, candidate_margins, specs=SPECS)
    ranking = ranking.rename(columns={'tuning_f1': 'selection_f1_optimistic', 'default_f1': 'fixed_threshold_oof_f1'})
    ranking.to_csv(OUT / 'candidate_comparison.csv', index=False)
    selected = ranking.iloc[0]
    oof = pd.DataFrame({'id': train.id, 'target': y, 'fold': fold_ids,
                        'nested_prediction': nested_prediction, 'nested_decision_score': nested_margins})
    for name, margin in candidate_margins.items():
        oof[f'{name}_margin'] = margin
    oof.to_csv(OUT / 'oof_predictions.csv', index=False)

    # Select final settings from training-only CV, then refit on every train row.
    test = clean_metadata(read_training_csv(RAW_DATA_DIR / 'test.csv'))
    template = pd.read_csv(RAW_DATA_DIR / 'sample_submission.csv')
    assert test.id.is_unique and template.id.is_unique
    assert set(test.id) == set(template.id) and len(test) == len(template)
    assert not set(train.id).intersection(test.id)
    pairs = feature_builder(X, test[['text', 'keyword']])
    spec = BY_NAME[selected.model]
    fit_matrix, test_matrix = pairs[spec['features']]
    test_margin = fit_decision_scores(spec, fit_matrix, y, test_matrix)
    ordered_margin = template.id.map(pd.Series(test_margin, index=test.id))
    assert ordered_margin.notna().all()
    submission = template.copy()
    submission['target'] = (ordered_margin >= selected.threshold_margin).astype('int64')
    submission.to_csv(SUBMISSIONS_DIR / submission_name, index=False)
    saved = pd.read_csv(SUBMISSIONS_DIR / submission_name)
    pd.testing.assert_frame_equal(saved, submission)
    assert saved.id.tolist() == template.id.tolist() and saved.target.isin([0, 1]).all()
    pd.DataFrame({'id': template.id, 'decision_score': ordered_margin,
                   'threshold_margin': float(selected.threshold_margin)}).to_csv(OUT / 'test_scores.csv', index=False)

    metrics = {'source_sha256': hashes, 'candidate_specs': SPECS,
                'threshold_grid_margins': THRESHOLDS.tolist(),
                'outer_folds': 5, 'inner_folds': 3, 'seed': SEED,
                'baseline_oof_metrics': baseline_scores, 'nested_oof_metrics': nested_scores,
                'baseline_mean_fold_f1': float(nested_frame.baseline_f1.mean()),
                'nested_mean_fold_f1': float(nested_frame.f1.mean()),
                'nested_std_fold_f1': float(nested_frame.f1.std(ddof=1)),
                'final_model': spec, 'final_threshold_margin': float(selected.threshold_margin),
                'final_training_cv_selection_f1_optimistic': float(selected.selection_f1_optimistic),
                'nested_confusion_matrix': confusion_matrix(y, nested_prediction, labels=[0, 1]).tolist(),
                'prediction_counts': {str(t): int(saved.target.eq(t).sum()) for t in [0, 1]},
                'submission_file': f'submissions/{submission_name}',
                'submission_sha256': hashlib.sha256((SUBMISSIONS_DIR / submission_name).read_bytes()).hexdigest(),
                'additional_context': extra_metadata or {},
                'adopted': bool(nested_scores['f1'] > baseline_scores['f1']),
                'submitted_to_kaggle': False, 'elapsed_seconds': time.perf_counter() - started}
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    comparison = pd.DataFrame([
        {'method': 'Original text + keyword, fixed threshold', **baseline_scores},
        {'method': 'Nested model/threshold search', **nested_scores},
    ])
    comparison.to_csv(OUT / 'honest_comparison.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(nested_frame.fold, nested_frame.baseline_f1, 'o-', label='Original baseline')
    ax.plot(nested_frame.fold, nested_frame.f1, 'o-', label='Nested search')
    ax.set(title='Same outer folds: positive-class F1', xlabel='Outer fold', ylabel='F1',
            xticks=range(1, 6), ylim=(0.65, 0.9))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'fold_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    report = f'''# Model improvement with nested grouped validation

## Measured comparison

{md_table(comparison)}

Nested mean fold F1: **{nested_frame.f1.mean():.4f}**, fold standard deviation **{nested_frame.f1.std(ddof=1):.4f}**. Original mean fold F1: **{nested_frame.baseline_f1.mean():.4f}**. Pooled out-of-fold F1 change: **{nested_scores['f1']-baseline_scores['f1']:+.4f}**.

Decision: {'adopt as the next submission candidate because nested validation improved' if metrics['adopted'] else 'do not adopt; nested validation did not improve on the original submission'}. Kaggle test performance has not been measured.

![Per-fold comparison](fold_comparison.png)

## Selection is separated from outer evaluation

The original five normalized-text-grouped folds are reused. For each outer training set, three additional grouped inner folds compare a predefined set of {len(SPECS)} models and {len(THRESHOLDS)} decision-margin thresholds. Model and threshold are selected using inner predictions only, then evaluated on the held-out outer fold. Every vectorizer, IDF value, categorical encoder, and supervised NB weight is fitted within its training subset. No normalized duplicate group crosses a boundary. Conflicting labels remain unchanged.

{md_table(nested_frame[['fold', 'selected_model', 'threshold_margin', 'f1', 'baseline_f1', 'delta_f1_vs_baseline']])}

Nested OOF performance evaluates the **selection procedure**, which can choose a different configuration in each outer fold. Final settings are subsequently selected by CV on all training rows. The final configuration's globally optimized selection score is optimistic and is not substituted for the nested score. Thresholds for SVMs are uncalibrated margins, not probabilities. Zero is the original decision threshold; for logistic regression, probability is sigmoid(margin).

## Candidate methods

The original TF-IDF + keyword logistic regression is included and reproduces the saved baseline predictions and probabilities.

{method_description}

{md_table(ranking)}

The `selection_f1_optimistic` column uses the same training-only OOF labels to choose model and threshold; it is a tuning score, not independent evaluation. `fixed_threshold_oof_f1` uses margin threshold zero but still involves comparing multiple candidates. Fold standard deviation is not a confidence interval. The original folds were used for prior model comparisons too; no new independent final test score is available.

## Final submission

Selected configuration: **{selected.model}**. Decision-margin threshold: **{selected.threshold_margin:.2f}**. Fit on all {len(train):,} training rows, predict all {len(test):,} test rows, and align to sample_submission.csv. Output: `../../submissions/{submission_name}`. The existing `../../submissions/submission.csv` is preserved. No test labels, additional disaster-labeled examples, or leaderboard feedback are used. No Kaggle upload has been performed.

## Why leaderboard scores can be 1.0

The [official Kaggle FAQ](https://www.kaggle.com/c/nlp-getting-started) says test labels are public and explains that this can produce perfect submissions. A perfect leaderboard score is therefore not evidence, by itself, of generalization to unseen tweets. This experiment uses the supplied training labels only. Local validation scores and Kaggle test scores are different evaluations.

## Reproduce

Run `{reproduce_command}` from the project root. `inner_search.csv`, `candidate_fold_metrics.csv`, `candidate_comparison.csv`, `nested_fold_metrics.csv`, and `oof_predictions.csv` preserve the search and predictions. `test_scores.csv` stores final decision scores; `metrics.json` records settings, hashes, prediction counts, and runtime.
'''
    (OUT / 'report.md').write_text(report, encoding='utf-8')
    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    print(comparison.to_string(index=False), flush=True)
    print(f'Final: {selected.model}, margin threshold={selected.threshold_margin:.2f}', flush=True)
    print(f'Saved {submission_name}; original inputs/submission unchanged. Runtime {metrics["elapsed_seconds"]:.1f}s.', flush=True)


if __name__ == '__main__':
    main()
