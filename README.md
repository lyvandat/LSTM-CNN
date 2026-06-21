# LSTM-CNN

Reproduction of a **Multi-Channel LSTM-CNN** model for Vietnamese sentiment
analysis, trained on the **UIT-VSFC** (Vietnamese Students' Feedback Corpus)
dataset. The model fuses a CNN branch and an LSTM branch over a shared word
embedding to classify each sentence as **negative / neutral / positive**.

Notebook: [`MultiChannel_LSTM_Reproduce_With_Flow.ipynb`](MultiChannel_LSTM_Reproduce_With_Flow.ipynb)
(Google Colab, GPU / T4).

---

## 1. Dataset

- **Source:** `uitnlp/vietnamese_students_feedback` (loaded from Hugging Face Hub
  parquet files: train / validation / test).
- The three original splits are concatenated into a single corpus and re-split
  with 5-fold cross-validation.

| Metric | Value |
| --- | --- |
| Total sentences | 16,175 |
| Class counts (neg / neu / pos) | 7,439 / 698 / 8,038 |

The dataset is highly imbalanced — the **neutral** class is tiny (~4%).

---

## 2. Preprocessing

Two tokenization settings are compared:

- **`without_token` (syllable level):** the text is kept as-is and split on
  whitespace.
- **`with_token` (word level):** [`pyvi`](https://pypi.org/project/pyvi/) word
  segmentation is run first, joining syllables into words
  (e.g. `sinh vien` → `sinh_vien`).

A vocabulary is built **per fold on the training split only** (`<pad>`, `<unk>`,
plus all tokens with `freq ≥ MIN_FREQ`). Sentences are encoded to integer IDs,
truncated / padded to `MAX_LEN = 50`.

---

## 3. Model architecture

A shared `nn.Embedding` feeds two parallel branches:

- **CNN branch** — 1-D convolutions with region sizes **3, 5, 7** (150 filters
  each), ReLU, max-over-time pooling, concatenated, then a dense layer to a
  100-dim feature.
- **LSTM branch** — a single-layer LSTM; the hidden state at the **last real
  word** (ignoring padding) is taken as a 128-dim feature.

The two feature vectors are concatenated and passed through a linear classifier
to produce the 3 class logits.

```
                    ┌──────────────► CNN (k=3,5,7) → max-pool → dense(100) ─┐
Embedding(200-d) ───┤                                                       ├─► concat → Linear → 3 classes
                    └──────────────► LSTM(128) → last-word hidden state ────┘
```

---

## 4. Training

| Hyperparameter | Value |
| --- | --- |
| Embedding dim | 200 |
| LSTM hidden | 128 |
| CNN filters / region sizes | 150 / (3, 5, 7) |
| CNN penultimate dim | 100 |
| Optimizer | Adamax |
| Learning rate | 0.002 |
| Loss | Cross-entropy |
| Batch size | 32 |
| Max epochs | 30 (early stopping, patience 3) |
| Folds | 5 (StratifiedKFold) |
| Max sequence length | 50 |
| Seed | 42 |

Early stopping monitors validation loss; the best-performing weights are
restored before evaluating each fold.

---

## 5. Results

### 5-fold cross-validation accuracy

| Setting | Accuracy (%) |
| --- | --- |
| Without token (syllable) | **90.70 ± 0.38** |
| With token (word segmentation) | 90.13 ± 0.23 |

The syllable-level setting slightly outperforms word segmentation on this
corpus.

### Per-class report — `without_token` (best setting)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| negative | 0.904 | 0.937 | 0.920 | 7,439 |
| neutral | 0.608 | 0.327 | 0.425 | 698 |
| positive | 0.923 | 0.930 | 0.926 | 8,038 |
| **accuracy** | | | **0.907** | 16,175 |
| macro avg | 0.812 | 0.731 | 0.757 | 16,175 |
| weighted avg | 0.901 | 0.907 | 0.902 | 16,175 |

The **neutral** class is the hardest (F1 ≈ 0.43) due to severe class imbalance,
while negative and positive are both classified well (F1 ≈ 0.92).

---

## 6. How to run

The notebook is self-contained and designed for Google Colab:

1. Open `MultiChannel_LSTM_Reproduce_With_Flow.ipynb` in Colab and select a GPU
   runtime (T4).
2. Run all cells in order. The first cells install dependencies:
   ```
   pyvi, huggingface_hub>=0.20, datasets
   ```
3. The data is downloaded automatically from the Hugging Face Hub; no manual
   download is required.
4. Training both settings end-to-end takes ~4 minutes on a T4 (~225s per
   setting).

Set `CONFIG["QUICK_TEST"] = True` for a faster sanity-check run with fewer folds
and epochs.

---

## 7. Notebook structure

| Section | Contents |
| --- | --- |
| 1–2 | Install packages, imports, config, seeding |
| 3 | Load UIT-VSFC from Hugging Face |
| 4 | Preprocessing, vocab, encoding (+ worked data-flow example) |
| 5 | Model definition (`CNNFeature`, `MultiChannelModel`) |
| 6 | Training loop with early stopping |
| 7 | Cross-validation harness |
| 8 | Run both tokenization settings |
| 9–10 | Accuracy table and per-class precision/recall/F1 |
