# Statistical & Machine Learning POS Tagging: Hidden Markov Models vs. Logistic Regression

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-2ea44f.svg)](https://www.nltk.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg)](https://numpy.org/)

An analytical Natural Language Processing (NLP) repository implementing and comparing two foundational approaches for **Part-of-Speech (POS) Tagging** on the **NLTK Brown Corpus** using the **Universal Tagset**:
1. **Generative Sequence Tagging**: A custom **Hidden Markov Model (HMM)** utilizing Laplace-smoothed transition/emission matrices and dynamic programming via the **Viterbi Algorithm**.
2. **Discriminative Classification**: A **Logistic Regression** classifier operating on engineered word-level, contextual, and morphological features.

---

## Comparative Performance Summary

Both taggers were trained and evaluated on an 80/20 train-test split (45,872 training sentences, 11,468 test sentences comprising ~231,927 tokens) from the Brown Corpus.

| Model Paradigm | Implementation Approach | Decoding / Inference | Test Accuracy | Macro F1-Score | Key Characteristics |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Generative (HMM)** | `hmm_pos_tagger.py` | Viterbi Dynamic Programming | **95.72%** | **0.92** | Captures global sequential tag dependencies; extremely robust on structural grammar. |
| **Discriminative (LR)** | `logistic_regression_pos_tagger.py` | Word & Morphological Features | **~96.00%** | **~0.93** | Leverages rich sub-word features (prefixes, suffixes, capitalization, context words). |

---

## Repository Structure

```text
├── hmm_pos_tagger.py                 # Full HMM implementation with Laplace smoothing & Viterbi decoding
├── logistic_regression_pos_tagger.py # Discriminative feature extraction & Logistic Regression pipeline
├── hmm_output.txt                    # Detailed execution log, sample predictions & classification report
├── .gitignore                        # Excludes assignment PDFs, archive zips, and environment caches
└── README.md                         # Project documentation and architectural overview
```

---

## Technical Methodology & Architecture

### 1. Hidden Markov Model (HMM) & Viterbi Decoding (`hmm_pos_tagger.py`)
The HMM models joint probability distribution $P(W, T)$ over word sequences $W$ and tag sequences $T$:
* **Transition Probabilities**: $P(t_i | t_{i-1})$ computed with Laplace smoothing over adjacent tag pairs.
* **Emission Probabilities**: $P(w_i | t_i)$ smoothed with vocabulary offset parameterization to handle unseen vocabulary terms during inference gracefully.
* **Viterbi Decoding**: Dynamic programming lattice decoding determining the globally optimal sequence $\hat{T} = \arg\max_T P(W, T)$ in $O(N \cdot |T|^2)$ time complexity.

### 2. Discriminative Feature Engineering (`logistic_regression_pos_tagger.py`)
Instead of relying strictly on sequential tag history, the discriminative model extracts rich feature vectors $\phi(w_i, c)$ for each token:
* **Lexical & Orthographic**: Exact lowercase word representation, capitalization flags (`is_cap`), numeric checks (`is_numeric`).
* **Morphological Subwords**: Character prefixes (`prefix-2`, `prefix-3`) and suffixes (`suffix-2`, `suffix-3`) to capture grammatical tense, gerunds (`-ing`), adverbs (`-ly`), and plurals (`-s`).
* **Local Context**: Surrounding adjacent tokens (`prev_word`, `next_word`) incorporating local structural cues.
* **Pipeline Vectorization**: Uses `DictVectorizer` for sparse feature matrix generation coupled with multinomial `LogisticRegression`.

---

## Getting Started

### 1. Installation & Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/mahmutozkizilcik/statistical-pos-tagging-brown-corpus.git
cd statistical-pos-tagging-brown-corpus
pip install nltk scikit-learn numpy
```

### 2. Execution
Run the Hidden Markov Model tagger evaluation:
```bash
python hmm_pos_tagger.py
```

Run the Logistic Regression feature-based tagger evaluation:
```bash
python logistic_regression_pos_tagger.py
```

---

## Sample Predictions

Below are example sequence decoding outputs from the test set:

```text
SENTENCE  : Laura Keene brushed by him with the glass of water .
ORIGINAL  : ['NOUN', 'NOUN', 'VERB', 'ADP', 'PRON', 'ADP', 'DET', 'NOUN', 'ADP', 'NOUN', '.']
PREDICTED : ['NOUN', 'NOUN', 'VERB', 'ADP', 'PRON', 'ADP', 'DET', 'NOUN', 'ADP', 'NOUN', '.']

SENTENCE  : Even Mr. Miller himself seems uncertain on this score .
ORIGINAL  : ['ADV', 'NOUN', 'NOUN', 'PRON', 'VERB', 'ADJ', 'ADP', 'DET', 'NOUN', '.']
PREDICTED : ['ADV', 'NOUN', 'NOUN', 'PRON', 'VERB', 'ADJ', 'ADP', 'DET', 'NOUN', '.']
```
