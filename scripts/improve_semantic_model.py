"""Evaluate frozen MPNet sentence features with nested grouped model selection."""
from disaster_tweets.paths import SEMANTIC_REPORTS_DIR

from disaster_tweets.search import main as run_search
from disaster_tweets.semantic_features import feature_pairs, prepare_embeddings

SPECS = [
    {'name': 'original_lr', 'features': 'original', 'classifier': 'lr', 'C': 1.0},
    *[{'name': f'embedding_lr_{c}', 'features': 'embedding', 'classifier': 'lr', 'C': float(c)}
      for c in [1, 4, 16, 64]],
    *[{'name': f'embedding_keyword_lr_{c}', 'features': 'embedding_keyword', 'classifier': 'lr', 'C': float(c)}
      for c in [1, 4, 16]],
    *[{'name': f'hybrid_lr_{c}', 'features': 'hybrid', 'classifier': 'lr', 'C': float(c)}
      for c in [.5, 1, 4]],
    *[{'name': f'embedding_svc_{c}', 'features': 'embedding', 'classifier': 'svc', 'C': float(c)}
      for c in [1, 4]],
]


if __name__ == '__main__':
    manifest = prepare_embeddings()
    description = f'''Frozen sentence embeddings come from [{manifest['model_id']}](https://huggingface.co/{manifest['model_id']}) at pinned revision `{manifest['revision']}`. The pretrained encoder is not fine-tuned on this competition. Its 768-dimensional outputs are L2-normalized. URL strings are removed, mentions are replaced with @user, HTML entities are unescaped, and whitespace is collapsed before encoding; case and hashtag words are preserved. Maximum sequence length is 128; {manifest['truncated_text_count']} of {manifest['unique_text_count']} unique texts are truncated.

The encoder computes each text independently without competition labels or learned corpus statistics. Its outputs can therefore be cached once for train/test; no TF-IDF, classifier, category mapping, or supervised statistics are fitted on validation/test rows. This uses a general pretrained model, not a competition-specific model with unknown validation overlap. The encoder's original pretraining corpus has not been audited for overlap with these texts.

- `embedding`: frozen sentence vectors alone.
- `embedding_keyword`: sentence vectors plus keyword one-hot features at weight .5; keyword categories are fitted within the training subset.
- `hybrid`: the original word/character TF-IDF + keyword representation, plus sentence vectors at weight 2. Original TF-IDF/keyword features are fitted within the training subset.
- Logistic regression: L2, liblinear, max_iter=1000, seed 42, C as recorded in the candidate table. LinearSVC: L2 squared-hinge, max_iter=10,000, seed 42. No class weighting. Locations and IDs are excluded.

This semantic family was attempted after the sparse search failed on the same outer folds. Nested validation separates tuning from each outer fold inside this predefined search, but adaptive model-family exploration on reused folds can still cause selection bias. A fresh independent evaluation is needed to confirm the final gain.'''
    run_search(feature_builder=feature_pairs, specs=SPECS,
               out_dir=SEMANTIC_REPORTS_DIR,
               submission_name='submission_improved.csv', method_description=description,
               extra_metadata={'embedding_manifest': manifest, 'prior_sparse_search_adopted': False},
               reproduce_command='python scripts/improve_semantic_model.py')
