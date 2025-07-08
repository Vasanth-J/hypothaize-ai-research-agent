# Hypothaize: An AI Research Assistant for Hypothesis Generation

Hypothaize is an AI-powered system that automatically generates research hypotheses from scientific paper abstracts. It blends advanced Transformer-based NLP models like T5 with traditional rule-based NLP++ techniques including keyword matching, pattern substitution, and decision trees. The tool assists researchers by providing semantically meaningful and novel hypotheses, helping to accelerate idea generation in academic writing.

##  Key Features

- Dual-engine architecture: Combines modern Transformer (T5) with traditional rule-based NLP++
- Uses the arXiv API to fetch research abstracts by category or keyword
- Text preprocessing using NLTK and SpaCy for tokenization, lemmatization, and NER
- Hypothesis generation using:
  - T5 Transformer model (fine-tuned for text generation)
  - Rule-based templates and keyword matching
  - Decision tree heuristic engine
- Evaluation metrics:
  - **BERTScore**, **semantic similarity**, **novelty**, **readability**, and **specificity**
- Visual comparison using radar and bar plots
- Streamlit-based user interface for interactive hypothesis generation


## How It Works

1. **Fetch abstracts** from arXiv using user-defined topics or keywords
2. **Preprocess text** (tokenization, lemmatization, NER)
3. **Generate hypotheses** using:
   - **T5 transformer** (prompt-based text generation)
   - **NLP++ rule engine** with templates and decision trees
4. **Evaluate** the hypotheses on multiple quality metrics
5. **Visualize** results using radar plots and comparative charts
