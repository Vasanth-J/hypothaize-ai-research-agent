"""
HypothAIze Project with NLP++ Comparative Analysis
=================================================
This enhanced version includes both traditional NLP techniques (NLP++) and modern deep learning
approaches for hypothesis generation, allowing for comparative analysis.
"""

import streamlit as st
import arxiv
import pandas as pd
import numpy as np
import torch
import re
import nltk
import random
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoModelForQuestionAnswering, AutoTokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import time

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    st.warning("NLTK resources could not be downloaded. Some features might be limited.")

# Load spaCy for rule-based NLP
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.warning("SpaCy model not loaded. Please install it with 'python -m spacy download en_core_web_sm'")
    nlp = None

# Setup page configuration
st.set_page_config(
    page_title="HypothAIze - AI-Driven Research Assistant with NLP++",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'papers_df' not in st.session_state:
    st.session_state.papers_df = None
if 'hypotheses' not in st.session_state:
    st.session_state.hypotheses = []
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

# ------ NLP++ TRADITIONAL APPROACHES -------

class NLPPlusPlus:
    """Class implementing traditional NLP approaches (NLP++)"""
    
    
    def __init__(self):
        """Initialize the NLP++ processor"""
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Domain-specific keywords for scientific hypothesis generation
        self.domain_keywords = {
            'cause_effect': ['cause', 'effect', 'impact', 'influence', 'affect', 'lead to', 'result in'],
            'comparison': ['compared to', 'more than', 'less than', 'higher', 'lower', 'increase', 'decrease'],
            'correlation': ['correlation', 'relationship', 'association', 'linked', 'connected', 'related to'],
            'methodology': ['method', 'approach', 'technique', 'protocol', 'procedure', 'experiment'],
            'mechanism': ['mechanism', 'pathway', 'process', 'function', 'system', 'interaction'],
            'outcome': ['outcome', 'result', 'finding', 'discovery', 'observation']
        }
        
        # Patterns for hypothesis generation
        self.hypothesis_patterns = [
            "KEYWORD_MECHANISM may influence KEYWORD_OUTCOME in the context of DOMAIN_CONCEPT.",
            "KEYWORD_COMPARISON between ENTITY_1 and ENTITY_2 could reveal insights about DOMAIN_CONCEPT.",
            "KEYWORD_CORRELATION between ENTITY_1 and DOMAIN_CONCEPT may suggest a new KEYWORD_METHODOLOGY.",
            "DOMAIN_CONCEPT could be enhanced by exploring the KEYWORD_MECHANISM of ENTITY_1.",
            "A novel KEYWORD_METHODOLOGY applying DOMAIN_CONCEPT might KEYWORD_CAUSE_EFFECT ENTITY_1.",
            "The KEYWORD_MECHANISM behind DOMAIN_CONCEPT could be explained through ENTITY_1 analysis."
        ]
        
    def preprocess_text(self, text):
        """Preprocess text using traditional NLP approaches"""
        # Lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    
    def extract_key_concepts(self, text, top_n=5):
        """Extract key concepts using TF-IDF"""
        # Create a small corpus with the single document
        corpus = [text]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=100)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get the most important terms
        scores = zip(feature_names, tfidf_matrix.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return [term for term, score in sorted_scores[:top_n]]
    
    def extract_named_entities(self, text):
        """Extract named entities using spaCy"""
        if nlp is None:
            # Fall back to regex-based extraction if spaCy is not available
            return self._extract_entities_regex(text)
        
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'LOC', 'PERSON', 'WORK_OF_ART']:
                entities.append((ent.text, ent.label_))
                
        return entities[:5]  # Limit to top 5 entities
    
    def _extract_entities_regex(self, text):
        """Fallback method for entity extraction using regex patterns"""
        # Simple regex patterns for capitalized terms that might be entities
        entity_pattern = r'\b[A-Z][a-zA-Z]+([ \-][A-Z][a-zA-Z]+)*\b'
        matches = re.findall(entity_pattern, text)
        return [(match, "UNKNOWN") for match in matches[:5]]
    
    def identify_domain(self, abstract):
        """Identify the scientific domain of the abstract"""
        # Simple keyword-based domain identification
        domains = {
            'computer_science': ['algorithm', 'computation', 'software', 'programming', 'neural network', 'machine learning'],
            'biology': ['gene', 'protein', 'cell', 'organism', 'molecular', 'enzyme', 'dna'],
            'physics': ['particle', 'quantum', 'relativity', 'energy', 'matter', 'wave', 'force'],
            'medicine': ['patient', 'disease', 'treatment', 'clinical', 'drug', 'therapy', 'diagnosis'],
            'chemistry': ['reaction', 'molecule', 'compound', 'element', 'acid', 'synthesis', 'catalyst']
        }
        
        # Convert abstract to lowercase for matching
        abstract_lower = abstract.lower()
        
        # Count domain keywords
        domain_scores = {}
        for domain, keywords in domains.items():
            score = sum(abstract_lower.count(keyword) for keyword in keywords)
            domain_scores[domain] = score
        
        # Get the domain with the highest score
        max_domain = max(domain_scores.items(), key=lambda x: x[1])
        
        if max_domain[1] > 0:
            return max_domain[0]
        else:
            return 'general_science'
    
    def extract_relations(self, text):
        """Extract subject-verb-object relations from text"""
        if nlp is None:
            return []
            
        doc = nlp(text)
        relations = []
        
        for sent in doc.sents:
            for token in sent:
                # Look for verbs
                if token.pos_ == "VERB":
                    # Find subject
                    subj = None
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subj = child.text
                            break
                    
                    # Find object
                    obj = None
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj"]:
                            obj = child.text
                            break
                    
                    if subj and obj:
                        relations.append((subj, token.text, obj))
        
        return relations[:5]  # Limit to top 5 relations
    
    def analyze_sentence_structure(self, text):
        """Analyze the syntactic structure of sentences in the text"""
        sentences = sent_tokenize(text)
        analysis = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)
            
            # Count parts of speech
            pos_counts = {}
            for _, tag in pos_tags:
                pos_counts[tag] = pos_counts.get(tag, 0) + 1
            
            # Calculate sentence complexity metrics
            complexity = {
                'length': len(tokens),
                'avg_word_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0,
                'pos_diversity': len(pos_counts) / len(tokens) if tokens else 0
            }
            
            analysis.append(complexity)
        
        # Calculate average complexity
        avg_complexity = {
            'avg_sentence_length': sum(a['length'] for a in analysis) / len(analysis) if analysis else 0,
            'avg_word_length': sum(a['avg_word_length'] for a in analysis) / len(analysis) if analysis else 0,
            'avg_pos_diversity': sum(a['pos_diversity'] for a in analysis) / len(analysis) if analysis else 0
        }
        
        return avg_complexity
    
    def generate_hypothesis_rule_based(self, abstract):
        """Generate a hypothesis using rule-based NLP++ approaches"""
        # Start timing
        start_time = time.time()
        
        # Extract key concepts from the abstract
        key_concepts = self.extract_key_concepts(abstract, top_n=7)
        
        # Extract named entities
        entities = self.extract_named_entities(abstract)
        entity_texts = [e[0] for e in entities]
        
        # Identify domain
        domain = self.identify_domain(abstract)
        
        # Extract relations
        relations = self.extract_relations(abstract)
        
        # Select a hypothesis pattern
        pattern = random.choice(self.hypothesis_patterns)
        
        # Replace placeholders with extracted elements
        for keyword_type, words in self.domain_keywords.items():
            placeholder = f"KEYWORD_{keyword_type.upper()}"
            if placeholder in pattern:
                pattern = pattern.replace(placeholder, random.choice(words))
        
        # Replace domain concept with a key concept
        if key_concepts:
            pattern = pattern.replace("DOMAIN_CONCEPT", random.choice(key_concepts))
        else:
            pattern = pattern.replace("DOMAIN_CONCEPT", "this area of research")
        
        # Replace entities
        if len(entity_texts) >= 2:
            pattern = pattern.replace("ENTITY_1", entity_texts[0])
            pattern = pattern.replace("ENTITY_2", entity_texts[1])
        elif len(entity_texts) == 1:
            pattern = pattern.replace("ENTITY_1", entity_texts[0])
            pattern = pattern.replace("ENTITY_2", "related factors")
        else:
            pattern = pattern.replace("ENTITY_1", "the primary factor")
            pattern = pattern.replace("ENTITY_2", "secondary factors")
        
        # Post-process to ensure proper grammar and capitalization
        hypothesis = pattern.strip()
        hypothesis = hypothesis[0].upper() + hypothesis[1:]
        
        # End timing
        processing_time = time.time() - start_time
        
        # Return hypothesis and metadata
        result = {
            "hypothesis": hypothesis,
            "method": "rule_based",
            "processing_time": processing_time,
            "metadata": {
                "key_concepts": key_concepts,
                "entities": entities,
                "domain": domain,
                "pattern_used": pattern
            }
        }
        
        return result
    
    def generate_hypothesis_decision_tree(self, abstract):
        """Generate a hypothesis using a decision tree approach"""
        # Start timing
        start_time = time.time()
        
        # Extract features from the abstract
        features = {}
        
        # Use complexity metrics
        complexity = self.analyze_sentence_structure(abstract)
        features.update(complexity)
        
        # Use term frequency for common scientific terms
        scientific_terms = [
            'analysis', 'approach', 'assessment', 'data', 'effect', 'evaluation', 
            'experiment', 'factor', 'method', 'model', 'process', 'research',
            'result', 'study', 'system', 'technique', 'test', 'theory'
        ]
        
        # Count term frequencies
        abstract_lower = abstract.lower()
        for term in scientific_terms:
            features[f'freq_{term}'] = abstract_lower.count(term)
        
        # Domain identification as a feature
        domain = self.identify_domain(abstract)
        
        # Decision tree logic (simplified) - in a real implementation this would be trained
        if domain == 'biology' or domain == 'medicine':
            if features['avg_sentence_length'] > 20:
                hypothesis_type = "mechanism"
            else:
                hypothesis_type = "correlation"
        elif domain == 'computer_science':
            if features['freq_model'] > 2:
                hypothesis_type = "performance"
            else:
                hypothesis_type = "methodology"
        elif domain == 'physics':
            hypothesis_type = "theoretical"
        else:
            hypothesis_type = "general"
        
        # Templates based on hypothesis type
        templates = {
            "mechanism": [
                "The mechanism of ACTION in SUBJECT may be mediated through the PATHWAY pathway.",
                "ACTION of SUBJECT might operate through a previously undescribed PATHWAY-dependent mechanism.",
                "A novel mechanism involving PATHWAY could explain how SUBJECT undergoes ACTION."
            ],
            "correlation": [
                "VARIABLE_A may be positively correlated with VARIABLE_B under CONDITIONS.",
                "An inverse relationship between VARIABLE_A and VARIABLE_B could exist when CONDITIONS are present.",
                "The correlation between VARIABLE_A and VARIABLE_B might vary depending on CONDITIONS."
            ],
            "performance": [
                "Optimizing PARAMETER could significantly improve the performance of METHOD for TASK.",
                "A novel variant of METHOD may outperform existing approaches for TASK by focusing on PARAMETER.",
                "Integrating METHOD_A with METHOD_B could enhance performance on TASK under CONDITIONS."
            ],
            "methodology": [
                "A revised METHOD that incorporates TECHNIQUE could improve outcomes for TASK.",
                "Combining TECHNIQUE_A and TECHNIQUE_B might yield a more effective METHOD for TASK.",
                "An alternative implementation of METHOD focusing on PARAMETER could be more efficient."
            ],
            "theoretical": [
                "CONCEPT_A and CONCEPT_B might be unified under a framework that considers PARAMETER.",
                "The apparent contradiction between CONCEPT_A and CONCEPT_B could be resolved by considering PARAMETER.",
                "A theoretical model incorporating both CONCEPT_A and PARAMETER might explain PHENOMENON."
            ],
            "general": [
                "Further investigation of SUBJECT may reveal important insights about ASPECT.",
                "The relationship between SUBJECT and ASPECT warrants more detailed examination.",
                "Understanding how SUBJECT affects ASPECT could have implications for APPLICATION."
            ]
        }
        
        # Select a template
        selected_template = random.choice(templates[hypothesis_type])
        
        # Extract key concepts to fill in the template
        key_concepts = self.extract_key_concepts(abstract, top_n=10)
        entities = self.extract_named_entities(abstract)
        entity_texts = [e[0] for e in entities]
        
        # Fill in template placeholders
        placeholders = {
            "ACTION": ["regulation", "expression", "activation", "inhibition", "modulation", "signaling"],
            "SUBJECT": entity_texts[:2] if entity_texts else key_concepts[:2],
            "PATHWAY": ["MAPK", "PI3K/AKT", "JAK/STAT", "Wnt", "Notch", "cellular"],
            "VARIABLE_A": key_concepts[:3] if key_concepts else ["the primary factor"],
            "VARIABLE_B": key_concepts[3:6] if len(key_concepts) > 3 else ["the outcome"],
            "CONDITIONS": ["specific conditions", "controlled environments", "certain parameters"],
            "PARAMETER": ["efficiency", "accuracy", "computational cost", "scalability", "robustness"],
            "METHOD": ["algorithm", "approach", "technique", "methodology", "framework"],
            "TASK": ["classification", "prediction", "optimization", "analysis", "processing"],
            "TECHNIQUE_A": ["machine learning", "statistical analysis", "neural networks", "optimization"],
            "TECHNIQUE_B": ["feature engineering", "data preprocessing", "ensemble methods", "regularization"],
            "CONCEPT_A": key_concepts[:2] if key_concepts else ["the established theory"],
            "CONCEPT_B": key_concepts[2:4] if len(key_concepts) > 2 else ["emerging observations"],
            "PHENOMENON": ["observed anomalies", "experimental results", "natural processes"],
            "ASPECT": key_concepts[4:7] if len(key_concepts) > 4 else ["related factors"],
            "APPLICATION": ["practical applications", "future research", "theoretical understanding"]
        }
        
        # Replace placeholders
        for placeholder, options in placeholders.items():
            if placeholder in selected_template:
                selected_value = random.choice(options) if isinstance(options, list) and options else placeholder.lower()
                selected_template = selected_template.replace(placeholder, selected_value)
        
        # End timing
        processing_time = time.time() - start_time
        
        # Return hypothesis and metadata
        result = {
            "hypothesis": selected_template,
            "method": "decision_tree",
            "processing_time": processing_time,
            "metadata": {
                "domain": domain,
                "hypothesis_type": hypothesis_type,
                "features": features,
                "key_concepts": key_concepts
            }
        }
        
        return result
    
    def generate_hypothesis_keyword_matching(self, abstract):
        """Generate a hypothesis using keyword matching approach"""
        # Start timing
        start_time = time.time()
        
        # Extract sentences from abstract
        sentences = sent_tokenize(abstract)
        
        # Score sentences based on keyword relevance
        sentence_scores = []
        
        # Keywords indicating potential for hypothesis formation
        hypothesis_keywords = [
            'suggest', 'indicate', 'imply', 'potential', 'possible', 'might', 'could',
            'hypothesis', 'theory', 'mechanism', 'relationship', 'correlation', 'association',
            'cause', 'effect', 'impact', 'influence', 'future', 'research', 'investigate'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(sentence_lower.count(keyword) for keyword in hypothesis_keywords)
            sentence_scores.append((sentence, score))
        
        # Sort sentences by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract top sentences
        top_sentences = [s[0] for s in sentence_scores[:2]]
        
        # Extract key concepts from top sentences
        combined_text = ' '.join(top_sentences)
        key_concepts = self.extract_key_concepts(combined_text, top_n=5)
        
        # Prepare hypothesis template
        prefix_templates = [
            "Based on these findings, we hypothesize that",
            "This suggests that",
            "These results indicate that",
            "A possible explanation is that",
            "We propose that"
        ]
        
        # Choose a random prefix
        prefix = random.choice(prefix_templates)
        
        # Take the most relevant sentence and modify it
        if top_sentences:
            base_sentence = top_sentences[0]
            
            # Transform sentence into hypothesis by removing specific phrases and patterns
            hypothesis_text = base_sentence
            
            # Remove phrases like "our study shows" or "we found that"
            patterns_to_remove = [
                r"our (study|research|analysis|results|findings)( \w+)* (show|demonstrate|indicate|suggest)s? that",
                r"we (found|discovered|observed|noted|showed|demonstrated) that",
                r"this (study|research|analysis|paper|work) (show|demonstrate|indicate|suggest)s? that",
                r"the results (show|demonstrate|indicate|suggest) that",
                r"(according to|based on) (our|the) (results|findings|analysis)"
            ]
            
            for pattern in patterns_to_remove:
                hypothesis_text = re.sub(pattern, "", hypothesis_text, flags=re.IGNORECASE)
            
            # If the sentence was stripped too much, use key concepts instead
            if len(hypothesis_text.strip()) < 30 or hypothesis_text == base_sentence:
                if len(key_concepts) >= 3:
                    hypothesis_text = f"{key_concepts[0]} may have a significant effect on {key_concepts[1]} through {key_concepts[2]}"
                elif len(key_concepts) >= 2:
                    hypothesis_text = f"{key_concepts[0]} may be directly related to {key_concepts[1]} under specific conditions"
                else:
                    hypothesis_text = f"the observed phenomenon may have important implications for future research"
                    
            # Combine prefix and modified sentence
            hypothesis = f"{prefix} {hypothesis_text.strip()}."
            
            # Ensure proper capitalization and ending with a period
            hypothesis = hypothesis[0].upper() + hypothesis[1:]
            if not hypothesis.endswith('.'):
                hypothesis += '.'
        else:
            # Fallback if no good sentences are found
            hypothesis = f"{prefix} further research in this area could yield important insights."
        
        # End timing
        processing_time = time.time() - start_time
        
        # Return hypothesis and metadata
        result = {
            "hypothesis": hypothesis,
            "method": "keyword_matching",
            "processing_time": processing_time,
            "metadata": {
                "top_sentences": top_sentences,
                "key_concepts": key_concepts,
                "sentence_scores": dict([(i, score) for i, (_, score) in enumerate(sentence_scores)])
            }
        }
        
        return result

# ------ DEEP LEARNING APPROACHES -------

@st.cache_resource
def load_hypothesis_model():
    """Load T5 model for hypothesis generation"""
    try:
        model_name = "t5-base"
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading T5 model: {e}")
        return None, None

@st.cache_resource
def load_embedding_model():
    """Load SentenceTransformer model for embeddings"""
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

def generate_hypothesis_transformer(abstract_text, model, tokenizer):
    """Generate a hypothesis using transformer-based deep learning"""
    # Start timing
    start_time = time.time()
    
    # Enhanced prompt with more specific guidance
    input_text = f"Based on the following scientific abstract, generate a novel, testable hypothesis that extends the research and identifies a specific knowledge gap: {abstract_text}"

    # Tokenize input with handling for long texts
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

    # Generate with more advanced parameters
    outputs = model.generate(
        **inputs,
        max_length=200,
        min_length=50,
        do_sample=True,
        temperature=0.9,
        top_p=0.92,
        top_k=50,
        num_return_sequences=1,
        no_repeat_ngram_size=3
    )

    hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the hypothesis
    hypothesis = hypothesis.replace("hypothesis:", "").strip()
    
    # End timing
    processing_time = time.time() - start_time
    
    # Return hypothesis and metadata
    result = {
        "hypothesis": hypothesis,
        "method": "transformer",
        "processing_time": processing_time,
        "metadata": {
            "model": "t5-base",
            "temperature": 0.9,
            "top_p": 0.92,
            "top_k": 50
        }
    }
    
    return result

# ------ EVALUATION METRICS -------

def calculate_bertscore(reference, hypothesis):
    """Calculate BERTScore between reference and hypothesis"""
    try:
        P, R, F1 = bert_score([hypothesis], [reference], lang="en")
        return {
            "precision": P.item(),
            "recall": R.item(),
            "f1": F1.item()
        }
    except Exception as e:
        st.error(f"Error calculating BERTScore: {e}")
        return {"precision": 0, "recall": 0, "f1": 0}

def calculate_semantic_similarity(text1, text2, embedding_model):
    """Calculate semantic similarity between two texts using embeddings"""
    try:
        embedding1 = embedding_model.encode([text1])[0]
        embedding2 = embedding_model.encode([text2])[0]
        similarity = util.cos_sim(
            torch.tensor([embedding1]), 
            torch.tensor([embedding2])
        ).item()
        return similarity
    except Exception as e:
        st.error(f"Error calculating semantic similarity: {e}")
        return 0.0

def evaluate_novelty(hypothesis, abstracts, embedding_model):
    """Evaluate how novel the hypothesis is compared to existing abstracts"""
    try:
        novelty_scores = []
        hypothesis_embedding = embedding_model.encode(hypothesis)
        
        for abstract in abstracts:
            abstract_embedding = embedding_model.encode(abstract)
            similarity = util.cos_sim(
                torch.tensor([hypothesis_embedding]), 
                torch.tensor([abstract_embedding])
            ).item()
            # Novelty is inverse of similarity
            novelty = 1 - similarity
            novelty_scores.append(novelty)
        
        return sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
    except Exception as e:
        st.error(f"Error evaluating novelty: {e}")
        return 0.0

def evaluate_readability(text):
    """Calculate readability metrics for the text"""
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    
    if not sentences:
        return {
            "words_per_sentence": 0,
            "avg_word_length": 0,
            "flesch_kincaid": 0
        }
    
    # Calculate basic metrics
    words_per_sentence = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Simple Flesch-Kincaid grade level approximation
    # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    # Here using word length as a rough proxy for syllables
    syllables_per_word = avg_word_length / 3
    flesch_kincaid = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    
    return {
        "words_per_sentence": words_per_sentence,
        "avg_word_length": avg_word_length,
        "flesch_kincaid": flesch_kincaid
    }

def evaluate_specificity(text):
    """Evaluate the specificity of the text using NLP features"""
    # Count specific word types that indicate specificity
    specificity_indicators = {
        'numbers': len(re.findall(r'\d+(?:\.\d+)?', text)),
        'precise_terms': len(re.findall(r'\b(specific|particular|exact|precise|definite)\b', text.lower())),
        'measurements': len(re.findall(r'\b\d+(?:\.\d+)?\s*(?:percent|mg|kg|ml|km|cm|mm|hz|db)\b', text.lower())),
        'comparison_terms': len(re.findall(r'\b(more|less|greater|fewer|higher|lower)\b', text.lower())),
        'scientific_terms': len(re.findall(r'\b(gene|protein|enzyme|neuron|pathway|receptor|molecule|ion|cell)\b', text.lower()))
    }
    
    # Calculate overall specificity score (simple sum for demo)
    specificity_score = sum(specificity_indicators.values()) / (len(text.split()) / 10)
    specificity_score = min(1.0, specificity_score)  # Cap at 1.0
    
    return {
        "specificity_score": specificity_score,
        "indicators": specificity_indicators
    }

# ------ ARXIV PAPER FETCHING -------

def fetch_arxiv_papers(category, query="", max_results=10):
    """Fetch papers from arXiv based on category and optional query"""
    search_query = f"cat:{category}"
    if query:
        search_query += f" AND all:{query}"
    
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for paper in search.results():
        paper_data = {
            "title": paper.title,
            "abstract": paper.summary,
            "authors": ", ".join(author.name for author in paper.authors),
            "published": paper.published.strftime("%Y-%m-%d"),
            "url": paper.pdf_url,
            "id": paper.entry_id.split('/')[-1]
        }
        papers.append(paper_data)
    
    return pd.DataFrame(papers) if papers else pd.DataFrame()

# ------ VISUALIZATION FUNCTIONS -------

def plot_evaluation_radar(metrics_dict):
    """Create a radar plot of evaluation metrics"""
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Ensure values are normalized between 0 and 1
    normalized_values = [min(max(v, 0), 1) for v in values]
    
    # Complete the loop for the radar chart
    values_with_closure = normalized_values + [normalized_values[0]]
    categories_with_closure = categories + [categories[0]]
    
    # Create figure and polar axes
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Set category locations
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_with_closure = angles + [angles[0]]
    
    # Plot values
    ax.plot(angles_with_closure, values_with_closure, 'o-', linewidth=2)
    ax.fill(angles_with_closure, values_with_closure, alpha=0.25)
    
    # Set category labels
    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    
    # Add value labels
    for angle, value, category in zip(angles, normalized_values, categories):
        ax.text(angle, value + 0.1, f"{value:.2f}", 
                ha='center', va='center', fontsize=9)
    
    # Set y-limits
    ax.set_ylim(0, 1.2)
    
    plt.title('Hypothesis Evaluation Metrics', size=15, y=1.1)
    
    return fig

def plot_comparison_bar(methods, metrics, title="Method Comparison"):
    """Create a bar chart comparing different methods on metrics"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    width = 0.8 / len(metrics)
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = width * i - width * len(metrics) / 2 + width / 2
        ax.bar(x + offset, values, width, label=metric_name)
    
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig

def plot_processing_times(methods, times):
    """Create a horizontal bar chart of processing times"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by time
    sorted_indices = np.argsort(times)
    sorted_methods = [methods[i] for i in sorted_indices]
    sorted_times = [times[i] for i in sorted_indices]
    
    ax.barh(sorted_methods, sorted_times, color='skyblue')
    ax.set_xlabel('Processing Time (seconds)')
    ax.set_title('Method Processing Time Comparison')
    
    # Add time labels
    for i, v in enumerate(sorted_times):
        ax.text(v + 0.05, i, f"{v:.3f}s", va='center')
    
    plt.tight_layout()
    return fig

# ------ MAIN APPLICATION -------

st.title("HypothAIze: AI-Driven Research Assistant with NLP++")
st.markdown("""
This application helps researchers generate and evaluate scientific hypotheses using both traditional NLP 
techniques (NLP++) and modern deep learning approaches. Compare different methods and see their strengths.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
pages = {
    "Home": "home",
    "Search Papers": "search",
    "Generate Hypotheses": "generate",
    "Compare Methods": "compare",
    "Compare NLP vs NLP++": "compare_nlp",
    "Chat with HypothAIze": "chat"
}

selected_page = st.sidebar.radio("Go to", list(pages.keys()))
st.session_state.current_page = pages[selected_page]

# Initialize models
with st.sidebar.expander("Model Status"):
    if 't5_model' not in st.session_state:
        st.session_state.t5_model, st.session_state.t5_tokenizer = load_hypothesis_model()
        st.write("✅ T5 Model" if st.session_state.t5_model else "❌ T5 Model")
    
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = load_embedding_model()
        st.write("✅ Embedding Model" if st.session_state.embedding_model else "❌ Embedding Model")
    
    if 'nlp_plus_plus' not in st.session_state:
        st.session_state.nlp_plus_plus = NLPPlusPlus()
        st.write("✅ NLP++ Initialized")

# Home Page
if st.session_state.current_page == "home":
    st.markdown("""
    ## Welcome to HypothAIze
    
    HypothAIze is an intelligent research assistant designed to help researchers generate novel scientific hypotheses.
    
    ### Key Features:
    - **Search papers** from arXiv and extract key information
    - **Generate hypotheses** using multiple approaches:
        - Traditional NLP (NLP++) techniques: Rule-based, Decision trees, Keyword matching
        - Deep learning: Transformer-based models
    - **Compare methods** to understand strengths and weaknesses
    - **Chat with HypothAIze** for research guidance and brainstorming
    
    ### Getting Started:
    1. Navigate to "Search Papers" to find relevant scientific literature
    2. Go to "Generate Hypotheses" to create new research ideas
    3. Use "Compare Methods" to evaluate different hypothesis generation approaches
    
    ### What makes this special?
    This tool uniquely combines both traditional NLP techniques and modern deep learning to show the relative 
    strengths of each approach and provide researchers with multiple perspectives.
    """)
    
    # Display a sample visualization
    st.markdown("### Sample Visualization of Method Comparison")
    
    sample_metrics = {
        "Novelty": [0.82, 0.65, 0.78, 0.91],
        "Specificity": [0.67, 0.89, 0.72, 0.63],
        "Readability": [0.85, 0.72, 0.79, 0.81]
    }
    
    sample_methods = ["Rule-based", "Decision Tree", "Keyword", "Transformer"]
    
    st.pyplot(plot_comparison_bar(sample_methods, sample_metrics))

# Search Papers Page
elif st.session_state.current_page == "search":
    st.header("Search Scientific Papers on arXiv")
    
    # Categories from arXiv
    arxiv_categories = {
        "Computer Science": "cs",
        "Mathematics": "math",
        "Physics": "physics",
        "Quantitative Biology": "q-bio",
        "Quantitative Finance": "q-fin",
        "Statistics": "stat",
        "Electrical Engineering": "eess",
        "Economics": "econ"
    }
    
    col1, col2 = st.columns(2)
    with col1:
        selected_category = st.selectbox("Select Category", list(arxiv_categories.keys()))
    with col2:
        search_query = st.text_input("Search Term (Optional)")
    
    fetch_button = st.button("Fetch Papers")
    
    if fetch_button:
        with st.spinner("Fetching papers from arXiv..."):
            category_code = arxiv_categories[selected_category]
            papers_df = fetch_arxiv_papers(category_code, search_query, max_results=15)
            st.session_state.papers_df = papers_df
    
    if st.session_state.papers_df is not None and not st.session_state.papers_df.empty:
        st.success(f"Found {len(st.session_state.papers_df)} papers.")
        
        # Display papers in a dataframe with sortable columns
        st.dataframe(
            st.session_state.papers_df[["title", "authors", "published"]],
            use_container_width=True,
            hide_index=True
        )
        
        # Inspect specific paper
        paper_indices = st.session_state.papers_df.index.tolist()
        selected_paper_idx = st.selectbox("Select a paper to inspect", paper_indices, 
                                         format_func=lambda x: st.session_state.papers_df.loc[x, "title"])
        
        if selected_paper_idx is not None:
            paper = st.session_state.papers_df.loc[selected_paper_idx]
            
            st.markdown(f"### {paper['title']}")
            st.markdown(f"**Authors:** {paper['authors']}")
            st.markdown(f"**Published:** {paper['published']}")
            st.markdown("**Abstract:**")
            st.markdown(f"{paper['abstract']}")
            
            # Add paper to hypothesis generation
            if st.button("Use for Hypothesis Generation"):
                if 'selected_papers' not in st.session_state:
                    st.session_state.selected_papers = []
                
                st.session_state.selected_papers.append({
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "id": paper["id"]
                })
                st.success(f"Added '{paper['title']}' for hypothesis generation!")

# Generate Hypotheses Page
elif st.session_state.current_page == "generate":
    st.header("Generate Scientific Hypotheses")
    
    if 'selected_papers' not in st.session_state or not st.session_state.selected_papers:
        st.warning("No papers selected yet. Please search and select papers first.")
    else:
        st.subheader("Selected Papers")
        for i, paper in enumerate(st.session_state.selected_papers):
            st.markdown(f"{i+1}. **{paper['title']}**")
        
        # Paper selection
        selected_paper_idx = st.selectbox(
            "Select paper to generate hypothesis for:",
            range(len(st.session_state.selected_papers)),
            format_func=lambda x: st.session_state.selected_papers[x]["title"]
        )
        
        paper = st.session_state.selected_papers[selected_paper_idx]
        
        st.markdown("#### Abstract:")
        st.markdown(paper["abstract"])
        
        # Method selection
        methods = {
            "NLP++ Rule-based": "rule_based",
            "NLP++ Decision Tree": "decision_tree",
            "NLP++ Keyword Matching": "keyword_matching",
            "Deep Learning Transformer": "transformer"
        }
        
        selected_methods = st.multiselect(
            "Select methods to generate hypotheses:",
            list(methods.keys()),
            default=["NLP++ Rule-based", "Deep Learning Transformer"]
        )
        
        if st.button("Generate Hypotheses") and selected_methods:
            with st.spinner("Generating hypotheses..."):
                all_hypotheses = []
                
                for method_name in selected_methods:
                    method_code = methods[method_name]
                    
                    if method_code == "rule_based":
                        result = st.session_state.nlp_plus_plus.generate_hypothesis_rule_based(paper["abstract"])
                    elif method_code == "decision_tree":
                        result = st.session_state.nlp_plus_plus.generate_hypothesis_decision_tree(paper["abstract"])
                    elif method_code == "keyword_matching":
                        result = st.session_state.nlp_plus_plus.generate_hypothesis_keyword_matching(paper["abstract"])
                    elif method_code == "transformer":
                        if st.session_state.t5_model and st.session_state.t5_tokenizer:
                            result = generate_hypothesis_transformer(
                                paper["abstract"], 
                                st.session_state.t5_model, 
                                st.session_state.t5_tokenizer
                            )
                        else:
                            st.error("Transformer model not loaded. Skipping this method.")
                            continue
                    
                    # Add metadata
                    result["paper_title"] = paper["title"]
                    result["method_name"] = method_name
                    
                    all_hypotheses.append(result)
                
                st.session_state.hypotheses.extend(all_hypotheses)
            
            st.success(f"Generated {len(all_hypotheses)} new hypotheses!")
        
        # Display existing hypotheses for this paper
        paper_hypotheses = [h for h in st.session_state.hypotheses if h["paper_title"] == paper["title"]]
        
        if paper_hypotheses:
            st.subheader("Generated Hypotheses")
            
            for i, hypothesis in enumerate(paper_hypotheses):
                with st.expander(f"{hypothesis['method_name']} - Hypothesis {i+1}"):
                    st.markdown(f"**Hypothesis:** {hypothesis['hypothesis']}")
                    st.markdown(f"**Generation Method:** {hypothesis['method_name']}")
                    st.markdown(f"**Processing Time:** {hypothesis['processing_time']:.3f} seconds")
                    
                    # Display method-specific metadata
                    if "metadata" in hypothesis:
                        with st.expander("Technical Details"):
                            for key, value in hypothesis["metadata"].items():
                                if isinstance(value, dict):
                                    st.json(value)
                                elif isinstance(value, list):
                                    st.write(f"{key}: {', '.join(str(v) for v in value)}")
                                else:
                                    st.write(f"{key}: {value}")
                    
                    # Evaluate hypothesis
                    if st.button(f"Evaluate Hypothesis {i+1}", key=f"eval_{i}"):
                        with st.spinner("Evaluating hypothesis..."):
                            # Only evaluate if embedding model is available
                            if st.session_state.embedding_model:
                                # Calculate metrics
                                readability = evaluate_readability(hypothesis["hypothesis"])
                                specificity = evaluate_specificity(hypothesis["hypothesis"])
                                
                                # Calculate novelty against the paper abstract
                                novelty = evaluate_novelty(
                                    hypothesis["hypothesis"], 
                                    [paper["abstract"]], 
                                    st.session_state.embedding_model
                                )
                                
                                # Prepare metrics for radar chart
                                metrics = {
                                    "Readability": 1 - (readability["flesch_kincaid"] / 20),  # Normalize
                                    "Specificity": specificity["specificity_score"],
                                    "Novelty": novelty,
                                    "Clarity": 0.5 + (1 - (readability["words_per_sentence"] / 40)) / 2,  # Normalize
                                    "Scientific Value": (specificity["specificity_score"] + novelty) / 2  # Combined metric
                                }
                                
                                # Plot radar chart
                                st.pyplot(plot_evaluation_radar(metrics))
                                
                                # Show detailed metrics
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Readability Metrics**")
                                    st.write(f"Words per sentence: {readability['words_per_sentence']:.2f}")
                                    st.write(f"Avg word length: {readability['avg_word_length']:.2f}")
                                    st.write(f"Flesch-Kincaid grade: {readability['flesch_kincaid']:.2f}")
                                
                                with col2:
                                    st.markdown("**Specificity Analysis**")
                                    st.write(f"Overall score: {specificity['specificity_score']:.2f}")
                                    for k, v in specificity['indicators'].items():
                                        st.write(f"{k}: {v}")
                            else:
                                st.error("Embedding model not loaded. Cannot evaluate hypothesis.")

# Compare Methods Page
elif st.session_state.current_page == "compare":
    st.header("Compare Hypothesis Generation Methods")
    
    if len(st.session_state.hypotheses) < 2:
        st.warning("Generate at least two hypotheses to compare methods.")
    else:
        # Group hypotheses by method
        methods_dict = {}
        for hypothesis in st.session_state.hypotheses:
            method = hypothesis["method_name"]
            if method not in methods_dict:
                methods_dict[method] = []
            methods_dict[method].append(hypothesis)
        
        st.subheader("Method Comparison")
        
        # Processing Time Comparison
        st.markdown("### Processing Time Comparison")
        
        methods = []
        times = []
        
        for method, hypotheses in methods_dict.items():
            methods.append(method)
            avg_time = sum(h["processing_time"] for h in hypotheses) / len(hypotheses)
            times.append(avg_time)
        
        st.pyplot(plot_processing_times(methods, times))
        
        # Quality metrics comparison if embedding model is available
        if st.session_state.embedding_model:
            st.markdown("### Quality Metrics Comparison")
            
            # Collect metrics for each method
            metrics_by_method = {
                "Specificity": [],
                "Readability": [],
                "Novelty": []
            }
            
            for method in methods:
                # Calculate average metrics for this method
                hypotheses = methods_dict[method]
                
                # Specificity
                avg_specificity = sum(evaluate_specificity(h["hypothesis"])["specificity_score"] 
                                     for h in hypotheses) / len(hypotheses)
                metrics_by_method["Specificity"].append(avg_specificity)
                
                # Readability (inverse of Flesch-Kincaid)
                avg_readability = 1 - sum(evaluate_readability(h["hypothesis"])["flesch_kincaid"] / 20
                                         for h in hypotheses) / len(hypotheses)
                metrics_by_method["Readability"].append(avg_readability)
                
                # Novelty (compared to paper abstracts)
                paper_abstracts = list(set(h.get("abstract", "") for h in st.session_state.hypotheses 
                                          if "abstract" in h))
                
                if not paper_abstracts:  # Fallback
                    paper_abstracts = [h["paper_title"] for h in hypotheses]
                    
                avg_novelty = sum(evaluate_novelty(h["hypothesis"], paper_abstracts, 
                                                  st.session_state.embedding_model)
                                 for h in hypotheses) / len(hypotheses)
                metrics_by_method["Novelty"].append(avg_novelty)
            
            # Plot comparison
            st.pyplot(plot_comparison_bar(methods, metrics_by_method, "Method Quality Comparison"))
            
            # Detailed comparison table
            st.markdown("### Detailed Comparison")
            
            comparison_data = []
            for i, method in enumerate(methods):
                comparison_data.append({
                    "Method": method,
                    "Avg Time (s)": f"{times[i]:.3f}",
                    "Specificity": f"{metrics_by_method['Specificity'][i]:.3f}",
                    "Readability": f"{metrics_by_method['Readability'][i]:.3f}",
                    "Novelty": f"{metrics_by_method['Novelty'][i]:.3f}",
                    "Sample Count": len(methods_dict[method])
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df)
            
            # Analysis
            st.markdown("### Analysis")
            
            best_time = methods[np.argmin(times)]
            best_specificity = methods[np.argmax(metrics_by_method["Specificity"])]
            best_readability = methods[np.argmax(metrics_by_method["Readability"])]
            best_novelty = methods[np.argmax(metrics_by_method["Novelty"])]
            
            st.markdown(f"""
            - **Fastest method:** {best_time}
            - **Most specific hypotheses:** {best_specificity}
            - **Most readable hypotheses:** {best_readability}
            - **Most novel hypotheses:** {best_novelty}
            
            #### Key Insights:
            
            Traditional NLP++ approaches tend to be faster but may lack the contextual understanding of 
            transformer models. Deep learning approaches generally produce more novel hypotheses but at 
            higher computational cost.
            
            For rapid hypothesis generation in familiar domains, rule-based approaches provide good 
            specificity with minimal processing time. For exploring new connections across disciplines,
            transformer models may discover insights that traditional methods miss.
            """)
        else:
            st.error("Embedding model not loaded. Cannot perform quality comparison.")
        
        # Example hypothesis comparison
        st.markdown("### Example Hypotheses by Method")
        
        for method, hypotheses in methods_dict.items():
            if hypotheses:
                with st.expander(f"{method} Example"):
                    # Show a random example
                    example = random.choice(hypotheses)
                    st.markdown(f"**Hypothesis:** {example['hypothesis']}")
                    st.markdown(f"**Paper:** {example['paper_title']}")
                    st.markdown(f"**Processing Time:** {example['processing_time']:.3f} seconds")

# Chat Page
elif st.session_state.current_page == "compare_nlp":
    import compare_nlp_vs_nlpplusplus

elif st.session_state.current_page == "chat":
    st.header("Chat with HypothAIze")

    
    st.markdown("""
    Discuss your research ideas and get intelligent recommendations from the HypothAIze assistant.
    The assistant can help you refine hypotheses, suggest research directions, and explain concepts.
    """)
    
    # Initialize chat history if needed
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    prompt = st.chat_input("Ask about your research...")
    
    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Simple but effective response generator
                if "hypothesis" in prompt.lower() or "idea" in prompt.lower():
                    # Extract NLP++ keywords if available
                    keywords = []
                    if st.session_state.nlp_plus_plus:
                        text = prompt + " " + " ".join([h["hypothesis"] for h in st.session_state.hypotheses[-3:]] 
                                                      if st.session_state.hypotheses else "")
                        keywords = st.session_state.nlp_plus_plus.extract_key_concepts(text)
                    
                    if keywords:
                        response = f"""Based on your interest in {', '.join(keywords[:3])}, here are some research directions to consider:
                        
1. Investigate how {keywords[0]} interacts with {keywords[1]} under different conditions
2. Explore the relationship between {keywords[1]} and {keywords[2]} in your specific domain
3. Consider developing a new methodology that combines approaches from both areas

I'd be happy to help you refine any of these directions into a more specific hypothesis."""
                    else:
                        response = """Here are some tips for developing a strong scientific hypothesis:

1. Make it testable and falsifiable
2. Be specific about the variables and relationships
3. Ground it in existing literature but look for gaps
4. Consider the practical implications and experimental design

Would you like help formulating a specific hypothesis in your field?"""
                
                elif "explain" in prompt.lower() or "what is" in prompt.lower() or "how does" in prompt.lower():
                    response = """I'd be happy to explain scientific concepts related to your research. For the best explanation, please:

1. Specify the exact concept you'd like explained
2. Let me know your background level (e.g., beginner, expert)
3. Mention if you're looking for mathematical details or an intuitive overview

This will help me provide the most useful explanation for your needs."""
                
                elif "method" in prompt.lower() or "compare" in prompt.lower() or "difference" in prompt.lower():
                    response = """The HypothAIze system uses multiple methods to generate hypotheses:

**Traditional NLP++ approaches:**
- Rule-based: Uses linguistic patterns and domain knowledge
- Decision tree: Adapts hypothesis structure based on paper characteristics
- Keyword matching: Extracts and reformulates key sentences

**Deep learning approaches:**
- Transformer models: Understand context and generate novel connections

Each has strengths - traditional methods are faster and more interpretable, while deep learning methods may find unexpected connections. The best approach depends on your specific research needs."""
                
                else:
                    response = """I'm your research assistant, and I can help with:

- Generating research hypotheses from scientific papers
- Refining your ideas into testable hypotheses
- Explaining scientific concepts and methodologies
- Suggesting research directions based on your interests
- Comparing different approaches to hypothesis generation

What aspect of your research would you like to discuss today?"""
                
                # Display response with typewriter effect
                full_response = ""
                message_placeholder = st.empty()
                for chunk in response.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)
                message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Add footer
st.markdown("---")
st.markdown("HypothAIze - Combining traditional NLP++ and modern deep learning for scientific hypothesis generation")
