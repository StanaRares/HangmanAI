# HangmanAI: Learning Decision Trees for Hangman

HangmanAI investigates how the best decision-tree strategy for Hangman changes with word length. The project builds a separate custom multiway decision tree for each cleaned English word-length bucket, then evaluates each tree against every word of that exact length.

The research question:

> If Hangman strategy is represented entirely as a decision tree, what first guess and follow-up policy maximizes win rate for 3-letter words, 4-letter words, 5-letter words, and longer English words?

This is not an algorithm-comparison app anymore. Random, entropy, random forest, boosting, and supervised sklearn decision-tree agents remain in the repository as legacy code, but the primary runtime player is the per-length serialized Hangman decision tree.

## Decision-Tree Model

The final player is a custom tree, not a classifier that imitates another solver.

Each internal node stores:

- the next guessed letter
- the current Hangman pattern
- guessed and incorrect letters
- remaining lives
- number of candidate words reaching the node
- child branches keyed by exact reveal pattern
- expected subtree win probability and mistake count

Hangman guesses are multiway splits. For a five-letter word, guessing `a` can branch to `00000`, `10000`, `01000`, `10001`, and many other position outcomes. Runtime play is therefore:

```text
public Hangman state
  -> load tree for word length
  -> current node says which letter to guess
  -> observe exact reveal pattern
  -> follow that branch
  -> repeat until solved or lives run out
```

No entropy agent, frequency agent, forest, boosting model, neural network, or reinforcement learner decides letters at inference time.

## Objective

The tree builder optimizes actual Hangman performance:

1. maximize win rate
2. minimize average incorrect guesses
3. minimize average total guesses
4. minimize expected tree depth
5. prefer smaller trees

The default mistake limit is 6 incorrect guesses. The implementation supports uniform weighting and wordfreq-prior weighting.

## Vocabulary

The research vocabulary is extracted from `wordfreq.iter_wordlist("en", wordlist="best")`. The pipeline does not impose a top-100,000 cap. Each token is cleaned by lowercasing, keeping ASCII `a-z` only, removing duplicates, excluding words shorter than 3 characters, rejecting obvious long-repeat artifacts, and requiring a vowel-like character for words longer than 3 characters.

Generated files:

- `data/processed/wordfreq_vocabulary.parquet`
- `data/processed/wordfreq_words.txt`
- `data/processed/by_length/words_<length>.txt`
- `data/processed/wordfreq_vocabulary_summary.csv`
- `data/processed/wordfreq_vocabulary_metadata.json`

Vocabulary extracted on this machine:

| Length | Words |
|---:|---:|
| 3 | 11,707 |
| 4 | 23,971 |
| 5 | 38,039 |
| 6 | 45,576 |
| 7 | 44,632 |
| 8 | 38,276 |
| 9 | 29,916 |
| 10 | 21,531 |
| 11 | 13,624 |
| 12 | 8,161 |
| 13 | 4,846 |
| 14 | 2,549 |
| 15 | 1,378 |
| 16 | 700 |
| 17 | 368 |
| 18 | 164 |
| 19 | 97 |
| 20 | 42 |
| 21 | 8 |
| 22 | 12 |
| 23 | 5 |
| 24 | 4 |
| 25 | 1 |
| 26 | 1 |
| 28 | 1 |
| 29 | 1 |
| 34 | 1 |

Total usable words: **285,611**.

## Architecture

```text
backend/app/tree/
  model.py          Multiway node, branch, objective, state-key, tree classes
  builder.py        Greedy and bounded-lookahead tree construction
  evaluation.py     Full-vocabulary tree evaluation by word length
  serialization.py  Compressed JSON tree save/load
  vocabulary.py     wordfreq extraction, cleaning, length buckets, frequency weights
backend/app/services/tree_service.py
  FastAPI service for tree loading, node reads, and game traversal
frontend/src/
  React decision-tree explorer, word-length dashboard, root-letter explorer
scripts/
  Reproducible vocabulary, training, evaluation, root analysis, weighting comparison
```

Serialized trees live under:

```text
models/
  length_3/uniform/tree.json.gz
  length_4/uniform/tree.json.gz
  ...
  length_34/uniform/tree.json.gz
```

Each length directory also stores metadata, vocabulary statistics, training statistics, and evaluation results.

## Results

These are real results from the generated `wordfreq` vocabulary and serialized uniform trees. They are best trees found under the recorded search profile and budget, not mathematical global optima.

| Length | Words | First | Win Rate | Avg Mistakes | Nodes | Leaves | Max Depth | Avg Depth | Strategy | Train s |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---|---:|
| 3 | 11,707 | A | 4.00% | 5.90 | 1,099 | 631 | 8 | 6.81 | greedy | 1.16 |
| 4 | 23,971 | A | 17.55% | 5.61 | 9,999 | 5,754 | 9 | 7.88 | greedy | 3.65 |
| 5 | 38,039 | A | 29.67% | 5.34 | 20,058 | 11,766 | 10 | 5.35 | greedy | 5.90 |
| 6 | 45,576 | A | 31.19% | 5.29 | 20,085 | 11,931 | 11 | 3.59 | greedy | 5.51 |
| 7 | 44,632 | E | 35.10% | 5.08 | 20,156 | 12,212 | 12 | 3.11 | greedy | 4.68 |
| 8 | 38,276 | E | 40.89% | 4.75 | 20,187 | 12,324 | 13 | 2.98 | greedy | 6.03 |
| 9 | 29,916 | E | 51.36% | 4.10 | 20,228 | 12,552 | 13 | 3.14 | greedy | 5.78 |
| 10 | 21,531 | E | 67.63% | 3.00 | 20,257 | 12,956 | 13 | 3.53 | greedy | 4.98 |
| 11 | 13,624 | E | 97.03% | 0.96 | 20,085 | 13,027 | 12 | 4.27 | greedy | 4.11 |
| 12 | 8,161 | I | 100.00% | 0.55 | 12,373 | 8,161 | 12 | 3.99 | greedy | 2.43 |
| 13 | 4,846 | I | 100.00% | 0.55 | 6,535 | 4,846 | 10 | 3.33 | bounded-lookahead | 117.42 |
| 14 | 2,549 | I | 100.00% | 0.45 | 3,404 | 2,549 | 10 | 3.04 | bounded-lookahead | 60.84 |
| 15 | 1,378 | I | 100.00% | 0.39 | 1,813 | 1,378 | 10 | 2.77 | bounded-lookahead | 32.19 |
| 16 | 700 | I | 100.00% | 0.28 | 915 | 700 | 8 | 2.43 | bounded-lookahead | 13.83 |
| 17 | 368 | I | 100.00% | 0.33 | 482 | 368 | 9 | 2.33 | bounded-lookahead | 6.82 |
| 18 | 164 | I | 100.00% | 0.22 | 198 | 164 | 3 | 1.79 | bounded-lookahead | 5.09 |
| 19 | 97 | R | 100.00% | 0.19 | 118 | 97 | 3 | 1.73 | bounded-lookahead | 2.91 |
| 20 | 42 | A | 100.00% | 0.10 | 52 | 42 | 3 | 1.45 | bounded-lookahead | 1.03 |
| 21 | 8 | L | 100.00% | 0.00 | 9 | 8 | 1 | 1.00 | bounded-lookahead | 0.15 |
| 22 | 12 | I | 100.00% | 0.25 | 14 | 12 | 2 | 1.25 | bounded-lookahead | 0.25 |
| 23 | 5 | R | 100.00% | 0.00 | 6 | 5 | 1 | 1.00 | bounded-lookahead | 0.08 |
| 24 | 4 | H | 100.00% | 0.00 | 5 | 4 | 1 | 1.00 | bounded-lookahead | 0.08 |
| 25 | 1 | deterministic | 100.00% | 0.00 | 1 | 1 | 0 | 0.00 | bounded-lookahead | 0.00 |
| 26 | 1 | deterministic | 100.00% | 0.00 | 1 | 1 | 0 | 0.00 | bounded-lookahead | 0.00 |
| 28 | 1 | deterministic | 100.00% | 0.00 | 1 | 1 | 0 | 0.00 | bounded-lookahead | 0.00 |
| 29 | 1 | deterministic | 100.00% | 0.00 | 1 | 1 | 0 | 0.00 | bounded-lookahead | 0.00 |
| 34 | 1 | deterministic | 100.00% | 0.00 | 1 | 1 | 0 | 0.00 | bounded-lookahead | 0.00 |

Generated research reports:

- `reports/results/performance_by_length.csv`
- `reports/results/best_first_letter_by_length.csv`
- `reports/results/tree_complexity.csv`
- `reports/results/hardest_words_by_length.csv`
- `reports/results/root_letter_analysis/`
- `reports/results/wordfreq_weighted/`
- `reports/results/weighting_comparison/`
- `reports/figures/win_rate_vs_word_length.png`
- `reports/figures/mistakes_vs_word_length.png`
- `reports/figures/tree_size_vs_word_length.png`
- `reports/figures/vocabulary_size_vs_word_length.png`
- `reports/figures/training_time_vs_word_length.png`

Root-letter analysis was computed for lengths 3, 4, and 5. Under the forced-root sweep, length 4 preferred `S`, length 5 preferred `A`, and length 3 had many roots tied on win rate and mistakes with `V` ahead by total guesses under the configured tie-breakers. The trained greedy length-3 tree itself starts with `A`.

## Frequency Weighting

Uniform weighting treats every cleaned vocabulary word as equally likely. Wordfreq weighting keeps the same rare words but changes split probabilities using Zipf-derived word weights.

The implemented secondary experiment currently includes a 5-letter wordfreq-weighted tree:

| Tree Weighting | First | Unweighted Win | Wordfreq-Weighted Win | Unweighted Mistakes | Weighted Mistakes | Nodes |
|---|:---:|---:|---:|---:|---:|---:|
| uniform | A | 29.67% | 34.92% | 5.34 | 5.21 | 20,058 |
| wordfreq | E | 27.38% | 69.82% | 5.41 | 3.73 | 20,048 |

This is an early but useful result: optimizing for common-word prior changed the 5-letter first guess from `A` to `E` and improved weighted performance, while reducing unweighted coverage.

## Commands

Install dependencies:

```bash
python -m pip install -r backend/requirements.txt
cd frontend
npm install
cd ..
```

Build the full wordfreq vocabulary:

```bash
python scripts/build_wordfreq_vocabulary.py --min-length 3
python scripts/vocabulary_summary.py
```

Train one uniform tree:

```bash
python scripts/train_tree.py --length 5
```

Train a bounded-lookahead tree:

```bash
python scripts/train_tree.py --length 5 --strategy optimized --lookahead 3
```

Train or resume all uniform lengths:

```bash
python scripts/train_all_trees.py --resume --profile auto --node-budget 20000 --checkpoint-every 1
```

Evaluate all uniform trees:

```bash
python scripts/evaluate_all_trees.py --weighting uniform
```

Run root-letter analysis:

```bash
python scripts/root_analysis.py --length 5 --profile fast --strategy greedy --node-budget 20000
```

Train and evaluate wordfreq-weighted trees:

```bash
python scripts/train_all_trees.py --weighting wordfreq --resume --profile auto --node-budget 20000 --checkpoint-every 1
python scripts/evaluate_all_trees.py --weighting wordfreq --output-dir reports/results/wordfreq_weighted
python scripts/compare_weightings.py --length 5
```

Continue root analysis for remaining lengths:

```bash
python scripts/root_analysis.py --length 6 --profile fast --strategy greedy --node-budget 20000
python scripts/root_analysis.py --length 7 --profile fast --strategy greedy --node-budget 20000
python scripts/root_analysis.py --length 8 --profile fast --strategy greedy --node-budget 20000
```

## API

Start the backend:

```bash
cd backend
uvicorn app.main:app --reload
```

Primary endpoints:

- `GET /health`
- `GET /vocabulary/summary`
- `GET /trees`
- `GET /trees/{length}`
- `GET /trees/{length}/summary`
- `GET /trees/{length}/root-analysis`
- `GET /trees/{length}/node/{node_id}`
- `POST /game/new`
- `POST /game/{game_id}/guess`
- `POST /game/{game_id}/tree-step`
- `POST /game/{game_id}/tree-play`
- `GET /results/by-length`
- `GET /results/hardest-words/{length}`

The API never sends an entire huge tree to the browser. The frontend lazily loads individual nodes and branch summaries.

## Frontend

Start the React app:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`.

The interface is organized around:

- Play: run a hidden word through the selected length tree and inspect each traversed node
- Tree Explorer: lazily inspect root and branch nodes
- Word-Length Dashboard: compare vocabulary size, first guesses, win rate, mistakes, nodes, depth, and strategy
- Root Letter Explorer: rank forced first-letter trees when root-analysis reports exist
- About: explain the research model for technical recruiters

## Docker

```bash
docker compose up --build
```

The backend uses:

```text
HANGMAN_VOCABULARY=/app/data/processed/wordfreq_vocabulary.parquet
HANGMAN_MODELS_DIR=/app/models
```

`docker compose config` has been verified for the current project configuration.

## Tests

```bash
pytest
cd frontend
npm run build
```

Test coverage includes:

- repeated-letter Hangman mechanics
- strict candidate filtering
- multiway decision-tree branches
- repeated-letter tree outcomes
- canonical state hashing for memoization
- tree traversal by observed reveal pattern
- full evaluation over a tiny vocabulary
- compressed serialization round trips
- strict per-length tree training
- FastAPI tree endpoints and gameplay

## Methodological Limits

These results should be read as best found under the recorded strategy and budget, not as globally proven optimal trees.

Important limits:

- Exact optimization was not attempted for the large wordfreq 3-, 4-, and 5-letter buckets.
- Several large buckets used greedy construction with a 20,000-node budget, so additional nodes or deeper lookahead may improve short-word win rates.
- `wordfreq` includes many obscure and short alphabetic tokens after cleaning; that makes short-word Hangman much harder than a curated classroom dictionary.
- Root-letter sweeps have currently been generated for lengths 3, 4, and 5 only.
- The frequency-weighted experiment has been run for length 5 only.
- The old sklearn/ensemble code remains as legacy implementation history and is not the main runtime model.

