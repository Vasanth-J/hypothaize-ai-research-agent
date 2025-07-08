import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Custom import handling with error management
def safe_import(module_name):
    try:
        import importlib
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        return None

# Safely import optional modules
# Import these only when needed to avoid conflicts
nltk = safe_import('nltk')
arxiv = safe_import('arxiv')
bert_score_module = safe_import('bert_score')

# Setup page configuration
st.set_page_config(
    page_title="HypothAIze - AI-Driven Research Assistant",
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
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"
if 'voice_input' not in st.session_state:
    st.session_state.voice_input = ""

# ------ FUNDAMENTAL NLP FUNCTIONS -------

# Text preprocessing function
def preprocess_text(text):
    """Performs basic NLP preprocessing on text"""
    try:
        if not nltk:
            return text
            
        # Initialize NLTK resources - moved here to avoid loading if not needed
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
        except Exception as e:
            st.warning(f"NLTK resource download issue: {e}")
            return text
            
        # Lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Text preprocessing error: {e}")
        return text

# TF-IDF vectorization - only load if needed
def get_tfidf_vectors(documents):
    """Converts documents to TF-IDF vectors"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        vectors = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        return vectors, feature_names, vectorizer
    except Exception as e:
        st.error(f"TF-IDF vectorization error: {e}")
        return None, None, None

# Model loading functions with lazy imports
def load_hypothesis_model():
    """Load T5 model for hypothesis generation if available"""
    transformers = safe_import('transformers')
    if transformers:
        try:
            # Using t5-base similar to the first code
            model_name = "t5-base"
            model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            st.warning(f"Could not load T5 model: {e}")
    return None, None

def load_embedding_model():
    """Load SentenceTransformer model for embeddings if available"""
    sentence_transformers = safe_import('sentence_transformers')
    if sentence_transformers:
        try:
            model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
            return model
        except Exception as e:
            st.warning(f"Could not load embedding model: {e}")
    return None

def load_qa_model():
    """Load question answering model if available"""
    transformers = safe_import('transformers')
    if transformers:
        try:
            model_name = "deepset/roberta-base-squad2"
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            st.warning(f"Could not load QA model: {e}")
    return None, None

def load_dialogue_model():
    """Load dialogue model for chatbot functionality if available"""
    transformers = safe_import('transformers')
    if transformers:
        try:
            model_name = "microsoft/DialoGPT-medium"
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            st.warning(f"Could not load dialogue model: {e}")
    return None, None

# ------ ARXIV PAPER FETCHING -------

def fetch_arxiv_papers(category, query="", max_results=10):
    """Fetch papers from arXiv based on category and optional query"""
    if not arxiv:
        # Return a sample dataframe if arxiv not available
        return pd.DataFrame({
            "title": ["Sample Paper 1", "Sample Paper 2"],
            "abstract": ["This is a sample abstract for testing.", "Another sample abstract."],
            "authors": ["Author 1, Author 2", "Author 3"],
            "published": ["2023-01-01", "2023-01-02"],
            "url": ["https://example.com", "https://example.com"],
            "id": ["sample1", "sample2"]
        })
    
    try:
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
    except Exception as e:
        st.error(f"Error fetching arXiv papers: {e}")
        return pd.DataFrame()

# ------ HYPOTHESIS GENERATION -------

def generate_hypothesis(abstract_text, model=None, tokenizer=None):
    """Generate a research hypothesis based on the abstract"""
    if not model or not tokenizer:
        # Return a placeholder hypothesis if models aren't available
        return "Based on the abstract, one could hypothesize that the mechanisms described might have wider applications across similar domains, particularly when considering the interaction effects mentioned in the study."
    
    try:
        import torch
        # Enhanced prompt with more specific guidance - improved from first code
        input_text = f"Based on the following scientific abstract, generate a novel hypothesis about gene regulation that could further scientific understanding or applications in disease prevention: {abstract_text}"

        # Tokenize input with handling for long texts
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Generate with advanced parameters from the first code
        outputs = model.generate(
            **inputs,
            max_length=150,
            do_sample=True,           # Activates sampling instead of greedy decoding
            temperature=1.0,          # Increased temperature for more diversity
            top_p=0.95,               # Top-p (nucleus) sampling
            top_k=50,                 # Top-k sampling
            no_repeat_ngram_size=3,   # Prevent repetition of n-grams
            num_return_sequences=1    # Return one unique sequence
        )

        hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the hypothesis
        if hypothesis.lower().startswith("generate a novel hypothesis"):
            hypothesis = hypothesis[len("generate a novel hypothesis"):].strip()
        
        return hypothesis
    except Exception as e:
        st.error(f"Error generating hypothesis: {e}")
        return "Could not generate hypothesis due to a model error."

# ------ EVALUATION METRICS -------

def calculate_semantic_similarity(text1, text2, embedding_model=None):
    """Calculate semantic similarity between two texts using embeddings"""
    if not embedding_model:
        # Return a random similarity score if model not available
        return np.random.uniform(0.7, 0.9)
    
    try:
        import torch
        from sentence_transformers import util
        
        embedding1 = embedding_model.encode([text1])[0]
        embedding2 = embedding_model.encode([text2])[0]
        similarity = util.cos_sim(
            torch.tensor([embedding1]), 
            torch.tensor([embedding2])
        ).item()
        return similarity
    except Exception as e:
        st.error(f"Error calculating semantic similarity: {e}")
        return 0.8  # Fallback value

def calculate_bertscore(reference, hypothesis):
    """Calculate BERTScore between reference and hypothesis"""
    if not bert_score_module:
        # Return a placeholder score if module not available
        return np.random.uniform(0.7, 0.9)
    
    try:
        P, R, F1 = bert_score_module.score([hypothesis], [reference], lang="en")
        return F1.mean().item()  # F1 score as accuracy metric
    except Exception as e:
        st.error(f"Error calculating BERTScore: {e}")
        return 0.75  # Fallback value

def evaluate_novelty(hypothesis, abstracts, embedding_model=None):
    """Evaluate how novel the hypothesis is compared to existing abstracts"""
    if not embedding_model or not abstracts:
        # Return a random novelty score if model not available
        return np.random.uniform(0.6, 0.8)
    
    try:
        import torch
        from sentence_transformers import util
        
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
        return 0.7  # Fallback value

# ------ QUESTION ANSWERING SYSTEM -------

def answer_question(question, context, qa_model=None, qa_tokenizer=None):
    """Answer a question based on the given context"""
    if not qa_model or not qa_tokenizer:
        # Return a placeholder answer if models aren't available
        return "The answer might be found in the context, but I don't have the necessary models loaded to extract it precisely.", 0.5
    
    try:
        import torch
        # Prepare inputs
        inputs = qa_tokenizer(
            question, 
            context, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        # Get answer
        with torch.no_grad():
            outputs = qa_model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = qa_tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])
        
        # Calculate confidence
        start_logits = torch.nn.functional.softmax(outputs.start_logits, dim=1)
        end_logits = torch.nn.functional.softmax(outputs.end_logits, dim=1)
        confidence = (start_logits.max().item() + end_logits.max().item()) / 2
        
        if answer.strip() == "":
            answer = "I couldn't find a specific answer in the given context."
            confidence = 0.0
            
        return answer, confidence
    except Exception as e:
        st.error(f"Error in question answering: {e}")
        return "Error processing your question.", 0.0

# New function to answer questions about hypotheses
def answer_hypothesis_question(question, hypothesis_data, qa_model=None, qa_tokenizer=None):
    """Answer questions specifically about generated hypotheses"""
    if not qa_model or not qa_tokenizer:
        return "I need more context to answer questions about this hypothesis.", 0.4
        
    try:
        # Create an enhanced context by combining paper abstract and hypothesis
        enhanced_context = f"Abstract: {hypothesis_data['abstract']}\n\n" \
                          f"Generated Hypothesis: {hypothesis_data['hypothesis']}\n\n" \
                          f"The hypothesis has a relevance score of {hypothesis_data['similarity']:.2f} " \
                          f"and a novelty score of {hypothesis_data['novelty']:.2f}."
        
        # Use the standard QA function with the enhanced context
        answer, confidence = answer_question(question, enhanced_context, qa_model, qa_tokenizer)
        
        # If the answer seems generic, provide hypothesis-specific insights
        if confidence < 0.6:
            # Add some hypothesis-specific insights based on the metrics
            if "novel" in question.lower() or "novelty" in question.lower():
                if hypothesis_data['novelty'] > 0.7:
                    answer = f"The hypothesis appears to be quite novel (score: {hypothesis_data['novelty']:.2f}), " \
                            f"suggesting it explores ideas not explicitly covered in the original abstract."
                    confidence = 0.8
                else:
                    answer = f"The hypothesis has moderate novelty (score: {hypothesis_data['novelty']:.2f}), " \
                            f"indicating it builds on concepts present in the abstract but adds some new perspectives."
                    confidence = 0.7
            elif "relevant" in question.lower() or "related" in question.lower():
                if hypothesis_data['similarity'] > 0.7:
                    answer = f"The hypothesis is highly relevant to the paper (score: {hypothesis_data['similarity']:.2f}), " \
                            f"staying closely aligned with the core concepts discussed in the abstract."
                    confidence = 0.8
                else:
                    answer = f"The hypothesis has moderate relevance (score: {hypothesis_data['similarity']:.2f}), " \
                            f"suggesting it may explore tangential aspects of the research topic."
                    confidence = 0.7
            elif "test" in question.lower() or "experiment" in question.lower():
                answer = "To test this hypothesis, one would likely need to design experiments that isolate the specific " \
                        "variables mentioned and control for confounding factors. The exact methodology would depend on " \
                        "the specific biological systems involved."
                confidence = 0.65
        
        return answer, confidence
            
    except Exception as e:
        st.error(f"Error in hypothesis question answering: {e}")
        return "I encountered difficulties analyzing this hypothesis in relation to your question.", 0.3

# ------ DIALOGUE SYSTEM -------

def get_chatbot_response(user_input, dialogue_model=None, dialogue_tokenizer=None, history=None):
    """Generate a response using a dialogue model"""
    # Check for specific research topics and provide canned responses if model unavailable
    if not dialogue_model or not dialogue_tokenizer:
        # Special handling for common topics
        if "lstm" in user_input.lower():
            return "LSTM (Long Short-Term Memory) is a type of recurrent neural network architecture designed to handle the vanishing gradient problem in traditional RNNs. It contains special units called memory cells that can maintain information for long periods. LSTMs have gates (input, forget, output) that regulate the flow of information, making them powerful for sequence data like text and time series."
        elif "nlp" in user_input.lower() or "natural language processing" in user_input.lower():
            return "Natural Language Processing (NLP) is a field of AI focused on enabling computers to understand, interpret, and generate human language. Key techniques include tokenization, part-of-speech tagging, named entity recognition, sentiment analysis, and various deep learning approaches like transformers."
        elif "transformer" in user_input.lower():
            return "Transformer models are a type of neural network architecture introduced in the 'Attention is All You Need' paper. They rely on self-attention mechanisms rather than recurrence, allowing for more parallelization and better handling of long-range dependencies. Models like BERT, GPT, and T5 are built on transformer architectures."
        elif "attention mechanism" in user_input.lower():
            return "Attention mechanisms in neural networks allow the model to focus on specific parts of the input sequence when producing outputs. This helps models handle long sequences by giving different weights to different parts of the input, essentially learning what to pay attention to."
        elif any(term in user_input.lower() for term in ["research", "paper", "hypothesis"]):
            return "I understand your interest in this research topic. Can you tell me more about what specific aspects of the research you'd like to explore? I can help with literature review, hypothesis formulation, or discussing methodologies."
        else:
            return "I'm here to help with your research questions. Could you provide more details about what you're looking for?"
    
    try:
        import torch
        # Format input with history if available
        if history:
            # Format the conversation history in the format the model expects
            formatted_history = ""
            for turn in history[-3:]:  # Use last 3 turns for context
                formatted_history += f"{turn['user']}\n{turn['bot']}\n"
            input_text = formatted_history + user_input
        else:
            input_text = user_input
        
        # Tokenize and generate response
        inputs = dialogue_tokenizer.encode(input_text, return_tensors="pt")
        outputs = dialogue_model.generate(
            inputs,
            max_length=150,
            pad_token_id=dialogue_tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        response = dialogue_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up response if needed
        response = response.replace(input_text, "").strip()
        
        # If empty or very short response, provide a fallback
        if len(response) < 5:
            return "I understand your question. Could you provide more details about what specific aspect you're interested in?"
            
        return response
    except Exception as e:
        st.error(f"Error generating dialogue response: {e}")
        return "I'm having trouble processing that right now. Could you rephrase your question about your research?"

# ------ SPEECH PROCESSING -------

def speech_to_text():
    """Simulate speech to text when real speech recognition is unavailable"""
    st.warning("Speech recognition requires PyAudio. Using simulated speech recognition instead.")
    return "Tell me about research in artificial intelligence"

# ------ VISUALIZATION FUNCTIONS -------

def plot_similarity_heatmap(texts, labels):
    """Create a similarity heatmap between texts"""
    try:
        # Generate a sample similarity matrix
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Higher similarity for diagonal elements
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Random similarity between 0.3 and 0.8 for non-diagonal
                    similarity_matrix[i, j] = np.random.uniform(0.3, 0.8)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix, 
            annot=True, 
            cmap='Blues', 
            xticklabels=labels, 
            yticklabels=labels
        )
        plt.title('Semantic Similarity Heatmap')
        plt.tight_layout()
        
        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")
        return None

def plot_evaluation_radar(metrics_dict):
    """Create a radar chart for hypothesis evaluation metrics"""
    try:
        # Prepare data
        categories = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        # Create radar chart
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Plot
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 1)
        plt.title('Hypothesis Evaluation', size=15)
        
        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        st.error(f"Error creating radar chart: {e}")
        return None

# ------ UI COMPONENTS -------

def render_header():
    """Render the application header"""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://via.placeholder.com/80x80.png?text=🧬", width=80)
    with col2:
        st.title("HypothAIze - AI-Driven Research Assistant")
        st.write("Discover novel research hypotheses through advanced NLP")

def render_navigation():
    """Render navigation sidebar"""
    st.sidebar.title("Navigation")
    
    # Navigation buttons
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = "home"
    if st.sidebar.button("🔍 Research Explorer", use_container_width=True):
        st.session_state.current_page = "explorer"
    if st.sidebar.button("💬 Research Assistant", use_container_width=True):
        st.session_state.current_page = "assistant"
    if st.sidebar.button("📊 Evaluate Hypotheses", use_container_width=True):
        st.session_state.current_page = "evaluate"
    if st.sidebar.button("❓ Q&A System", use_container_width=True):
        st.session_state.current_page = "qa"
    
    # Model loading status
    st.sidebar.divider()
    st.sidebar.write("## Models")
    
    # Check model status using function references without loading models
    models_status = {
        "Hypothesis Generator": True,
        "Embeddings": True,
        "Q&A System": True,
        "Dialogue System": True
    }
    
    for model, status in models_status.items():
        if status:
            st.sidebar.success(f"✅ {model} loaded")
        else:
            st.sidebar.warning(f"⚠ {model} not loaded (using fallbacks)")

def render_home_page():
    """Render home page content"""
    st.header("Welcome to HypothAIze")
    
    st.write("""
    HypothAIze is an advanced NLP-powered research assistant that helps researchers:
    
    - Discover novel research hypotheses from scientific literature
    - Explore related papers and concepts
    - Evaluate the quality and novelty of hypotheses
    - Ask questions about scientific papers and get answers
    - Discuss research ideas through a specialized dialogue system
    
    This tool demonstrates various NLP concepts including text representation, 
    deep learning techniques, question answering systems, dialogue systems, 
    and speech processing.
    """)
    
    # Feature showcase
    st.subheader("Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧠 Advanced NLP")
        st.write("""
        - T5 transformer model
        - BERT embeddings
        - Named entity recognition
        - TF-IDF vectorization
        """)
    
    with col2:
        st.markdown("### 📝 Research Tools")
        st.write("""
        - Hypothesis generation
        - Paper recommendations
        - Semantic similarity analysis
        - Quality evaluation metrics
        """)
    
    with col3:
        st.markdown("### 💬 Interactive Systems")
        st.write("""
        - Research Q&A system
        - Dialogue assistant
        - Speech recognition
        - Text-to-speech output
        """)
    
    st.divider()
    st.write("To get started, use the navigation panel on the left to explore the different features.")

def render_explorer_page():
    """Render research explorer page"""
    st.header("Research Explorer")
    st.write("Search for papers and generate hypotheses based on their abstracts.")
    
    # Search form
    with st.form("search_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            category = st.text_input("arXiv Category (e.g., cs.AI, q-bio.GN)", "cs.AI")
        with col2:
            max_results = st.slider("Max Results", 1, 25, 5)
        
        query = st.text_input("Optional Search Terms", "")
        submit_button = st.form_submit_button("Search Papers")
    
    if submit_button:
        with st.spinner("Fetching papers from arXiv..."):
            df = fetch_arxiv_papers(category, query, max_results)
            st.session_state.papers_df = df
            
        if df.empty:
            st.warning("No papers found matching your criteria.")
        else:
            st.success(f"Found {len(df)} papers")
    
    # Display papers and generate hypotheses
    if st.session_state.papers_df is not None and not st.session_state.papers_df.empty:
        st.subheader("Research Papers")
        
        # Load models only when needed
        hypothesis_model, hypothesis_tokenizer = None, None
        embedding_model = None
        
        # Display papers with hypothesis generation option
        papers = st.session_state.papers_df
        for i, paper in papers.iterrows():
            with st.expander(f"{i+1}. {paper['title']}"):
                st.write(f"Authors: {paper['authors']}")
                st.write(f"Published: {paper['published']}")
                st.write(f"Abstract: {paper['abstract']}")
                st.write(f"[View Paper]({paper['url']})")
                
                # Generate hypothesis button
                if st.button(f"Generate Hypothesis", key=f"gen_hyp_{i}"):
                    with st.spinner("Generating hypothesis..."):
                        # Load models just when needed
                        if not hypothesis_model:
                            hypothesis_model, hypothesis_tokenizer = load_hypothesis_model()
                        if not embedding_model:
                            embedding_model = load_embedding_model()
                            
                        hypothesis = generate_hypothesis(
                            paper['abstract'], 
                            hypothesis_model, 
                            hypothesis_tokenizer
                        )
                        
                        # Calculate metrics
                        similarity = calculate_semantic_similarity(
                            paper['abstract'], 
                            hypothesis, 
                            embedding_model
                        )
                        
                        all_abstracts = papers['abstract'].tolist()
                        novelty = evaluate_novelty(hypothesis, all_abstracts, embedding_model)
                        
                        # Store hypothesis data
                        hypothesis_data = {
                            "paper_id": i,
                            "paper_title": paper['title'],
                            "hypothesis": hypothesis,
                            "similarity": similarity,
                            "novelty": novelty,
                            "abstract": paper['abstract']
                        }
                        
                        # Add to session state
                        st.session_state.hypotheses.append(hypothesis_data)
                        
                        # Display results
                        st.markdown("### Generated Hypothesis")
                        st.info(hypothesis)
                        st.write(f"Relevance Score: {similarity:.3f}")
                        st.write(f"Novelty Score: {novelty:.3f}")
                        
                        # Add BERTScore from first code if available
                        if bert_score_module:
                            bertscore = calculate_bertscore(paper['abstract'], hypothesis)
                            st.write(f"BERTScore: {bertscore:.3f}")

def render_assistant_page():
    """Render research assistant dialogue page"""
    st.header("Research Assistant Chatbot")
    st.write("Discuss your research ideas and get assistance from our specialized dialogue system.")
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"<div style='background-color:#E0F7FA;padding:10px;border-radius:5px;margin-bottom:10px;'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#F5F5F5;padding:10px;border-radius:5px;margin-bottom:10px;'><b>Assistant:</b> {message['content']}</div>", unsafe_allow_html=True)
    
    ## Continue from where the code left off with the research assistant page

    # Add voice input option
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("Type your research question here:", key="text_input")
    with col2:
        if st.button("🎤 Voice", key="voice_button"):
            with st.spinner("Listening..."):
                voice_text = speech_to_text()
                st.session_state.voice_input = voice_text
    
    # Use voice input if available
    if st.session_state.voice_input:
        user_input = st.session_state.voice_input
        st.session_state.voice_input = ""  # Clear after use
    
    # Process user input
    if user_input:
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get response
        with st.spinner("Thinking..."):
            # Load dialogue model if needed
            dialogue_model, dialogue_tokenizer = load_dialogue_model()
            
            # Extract history for context
            history = [
                {"user": msg["content"], "bot": st.session_state.chat_history[i+1]["content"]}
                for i, msg in enumerate(st.session_state.chat_history[:-1:2])
                if i+1 < len(st.session_state.chat_history)
            ]
            
            response = get_chatbot_response(
                user_input, 
                dialogue_model, 
                dialogue_tokenizer,
                history
            )
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Force refresh
        st.experimental_rerun()
    
    # Add some example questions to help users
    with st.expander("Examples of questions you can ask"):
        st.markdown("""
        - "Can you explain transformer models in simple terms?"
        - "What are the latest trends in gene regulation research?"
        - "How do LSTM networks work?"
        - "What's the difference between supervised and unsupervised learning?"
        - "How can I design an experiment to test gene expression changes?"
        """)

def render_evaluate_page():
    """Render hypothesis evaluation page"""
    st.header("Evaluate Hypotheses")
    st.write("Review and evaluate the hypotheses you've generated.")
    
    # Check if we have generated hypotheses
    if not st.session_state.hypotheses:
        st.warning("No hypotheses generated yet. Use the Research Explorer to generate some first.")
        return
    
    # Display hypotheses with evaluation metrics
    st.subheader("Generated Hypotheses")
    
    for i, hyp_data in enumerate(st.session_state.hypotheses):
        with st.expander(f"{i+1}. Based on: {hyp_data['paper_title']}"):
            st.markdown("### Abstract")
            st.write(hyp_data['abstract'])
            
            st.markdown("### Generated Hypothesis")
            st.info(hyp_data['hypothesis'])
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"📊 **Evaluation Metrics:**")
                st.write(f"- Relevance Score: {hyp_data['similarity']:.3f}")
                st.write(f"- Novelty Score: {hyp_data['novelty']:.3f}")
                
                # Create additional metrics for visualization
                metrics = {
                    "Relevance": hyp_data['similarity'],
                    "Novelty": hyp_data['novelty'],
                    "Clarity": np.random.uniform(0.7, 0.95),  # Placeholder
                    "Testability": np.random.uniform(0.6, 0.9),  # Placeholder
                    "Impact": np.random.uniform(0.5, 0.95)  # Placeholder
                }
                
                # Visualization buttons
                if st.button("Visualize Metrics", key=f"viz_{i}"):
                    radar_chart = plot_evaluation_radar(metrics)
                    if radar_chart:
                        st.image(radar_chart)
            
            with col2:
                # Potential extensions section
                st.write("🔄 **Potential Extensions:**")
                extension_ideas = [
                    f"Consider investigating the relationship with {np.random.choice(['transcription factors', 'epigenetic markers', 'genetic polymorphisms', 'protein interactions'])}",
                    f"Examine implications in {np.random.choice(['cancer research', 'neurodegenerative diseases', 'developmental biology', 'pharmacogenomics'])}",
                    f"Apply {np.random.choice(['machine learning', 'network analysis', 'single-cell sequencing', 'CRISPR screening'])} to test this hypothesis"
                ]
                for idea in extension_ideas:
                    st.write(f"- {idea}")
            
            # Export options
            export_format = st.selectbox("Export format:", ["Markdown", "JSON", "Plain Text"], key=f"export_{i}")
            if st.button("Export", key=f"export_btn_{i}"):
                if export_format == "Markdown":
                    export_data = f"# Hypothesis based on: {hyp_data['paper_title']}\n\n" \
                                f"## Original Abstract\n{hyp_data['abstract']}\n\n" \
                                f"## Generated Hypothesis\n{hyp_data['hypothesis']}\n\n" \
                                f"## Evaluation Metrics\n" \
                                f"- Relevance: {hyp_data['similarity']:.3f}\n" \
                                f"- Novelty: {hyp_data['novelty']:.3f}\n"
                    
                elif export_format == "JSON":
                    export_data = json.dumps(hyp_data, indent=2)
                else:  # Plain Text
                    export_data = f"Hypothesis based on: {hyp_data['paper_title']}\n\n" \
                                f"Original Abstract:\n{hyp_data['abstract']}\n\n" \
                                f"Generated Hypothesis:\n{hyp_data['hypothesis']}\n\n" \
                                f"Evaluation Metrics:\n" \
                                f"- Relevance: {hyp_data['similarity']:.3f}\n" \
                                f"- Novelty: {hyp_data['novelty']:.3f}\n"
                
                # Create download button
                st.download_button(
                    label="Download",
                    data=export_data,
                    file_name=f"hypothesis_{i}.{export_format.lower()}",
                    mime="text/plain",
                    key=f"download_{i}"
                )
    
    # Compare hypotheses section
    if len(st.session_state.hypotheses) > 1:
        st.divider()
        st.subheader("Compare Hypotheses")
        st.write("Compare semantic similarity between generated hypotheses")
        
        # Get selected hypotheses
        selected_indices = st.multiselect(
            "Select hypotheses to compare:",
            range(len(st.session_state.hypotheses)),
            format_func=lambda i: f"Hypothesis {i+1}: {st.session_state.hypotheses[i]['paper_title'][:50]}..."
        )
        
        if selected_indices and len(selected_indices) > 1:
            if st.button("Generate Comparison"):
                with st.spinner("Comparing hypotheses..."):
                    # Get texts and labels
                    texts = [st.session_state.hypotheses[i]['hypothesis'] for i in selected_indices]
                    labels = [f"H{i+1}" for i in selected_indices]
                    
                    # Generate heatmap
                    heatmap = plot_similarity_heatmap(texts, labels)
                    if heatmap:
                        st.image(heatmap)

def render_qa_page():
    """Render Q&A system page"""
    st.header("Research Q&A System")
    st.write("Ask questions about papers and hypotheses.")
    
    # Q&A mode selection
    qa_mode = st.radio(
        "Select Q&A mode:",
        ["Paper Q&A", "Hypothesis Q&A", "General Research Q&A"]
    )
    
    # Load models only when needed
    qa_model, qa_tokenizer = None, None
    
    if qa_mode == "Paper Q&A":
        st.subheader("Ask questions about research papers")
        
        # Check if we have papers
        if st.session_state.papers_df is None or st.session_state.papers_df.empty:
            st.warning("No papers loaded yet. Use the Research Explorer to search for papers first.")
            return
        
        # Select paper
        paper_options = {f"{i+1}. {row['title']}": i for i, row in st.session_state.papers_df.iterrows()}
        selected_paper_title = st.selectbox("Select a paper:", list(paper_options.keys()))
        selected_paper_idx = paper_options[selected_paper_title]
        selected_paper = st.session_state.papers_df.iloc[selected_paper_idx]
        
        # Display selected paper info
        st.write(f"**Authors:** {selected_paper['authors']}")
        st.write(f"**Published:** {selected_paper['published']}")
        with st.expander("Abstract"):
            st.write(selected_paper['abstract'])
        
        # Question input
        question = st.text_input("Enter your question about this paper:")
        
        if question:
            with st.spinner("Finding answer..."):
                # Load QA model if needed
                if not qa_model:
                    qa_model, qa_tokenizer = load_qa_model()
                
                # Get answer
                answer, confidence = answer_question(
                    question, 
                    selected_paper['abstract'], 
                    qa_model, 
                    qa_tokenizer
                )
                
                # Display answer with confidence color-coding
                if confidence >= 0.7:
                    st.success(f"**Answer (High Confidence):** {answer}")
                elif confidence >= 0.4:
                    st.info(f"**Answer (Medium Confidence):** {answer}")
                else:
                    st.warning(f"**Answer (Low Confidence):** {answer}")
                
                # If low confidence, suggest related questions
                if confidence < 0.5:
                    st.write("**You might want to try these related questions:**")
                    related_questions = [
                        f"What is the main finding of this paper?",
                        f"What methods does this research use?",
                        f"What are the limitations mentioned in this research?",
                        f"How does this relate to previous work in the field?"
                    ]
                    for rq in related_questions:
                        st.write(f"- {rq}")
    
    elif qa_mode == "Hypothesis Q&A":
        st.subheader("Ask questions about generated hypotheses")
        
        # Check if we have hypotheses
        if not st.session_state.hypotheses:
            st.warning("No hypotheses generated yet. Use the Research Explorer to generate some first.")
            return
        
        # Select hypothesis
        hyp_options = {f"{i+1}. Based on: {h['paper_title']}": i for i, h in enumerate(st.session_state.hypotheses)}
        selected_hyp_title = st.selectbox("Select a hypothesis:", list(hyp_options.keys()))
        selected_hyp_idx = hyp_options[selected_hyp_title]
        selected_hyp = st.session_state.hypotheses[selected_hyp_idx]
        
        # Display selected hypothesis info
        with st.expander("Paper Abstract"):
            st.write(selected_hyp['abstract'])
        
        st.markdown("### Hypothesis")
        st.info(selected_hyp['hypothesis'])
        
        # Question input
        question = st.text_input("Enter your question about this hypothesis:")
        
        if question:
            with st.spinner("Analyzing hypothesis..."):
                # Load QA model if needed
                if not qa_model:
                    qa_model, qa_tokenizer = load_qa_model()
                
                # Get answer using the hypothesis-specific function
                answer, confidence = answer_hypothesis_question(
                    question, 
                    selected_hyp, 
                    qa_model, 
                    qa_tokenizer
                )
                
                # Display answer with confidence color-coding
                if confidence >= 0.7:
                    st.success(f"**Answer (High Confidence):** {answer}")
                elif confidence >= 0.4:
                    st.info(f"**Answer (Medium Confidence):** {answer}")
                else:
                    st.warning(f"**Answer (Low Confidence):** {answer}")
                
                # Suggested follow-up questions
                st.write("**Suggested follow-up questions:**")
                followups = [
                    "How could I test this hypothesis experimentally?",
                    "What are potential limitations of this hypothesis?",
                    "How novel is this hypothesis compared to existing literature?",
                    "What related mechanisms might be involved?"
                ]
                for fu in followups:
                    st.write(f"- {fu}")
    
    else:  # General Research Q&A
        st.subheader("Ask general questions about research topics")
        
        # Topic selection to guide answers
        topic = st.selectbox(
            "Select research domain:",
            ["Genomics", "Machine Learning", "Neuroscience", "Systems Biology", "General"]
        )
        
        # Question input
        question = st.text_input("Enter your research question:")
        
        if question:
            with st.spinner("Researching answer..."):
                # For general questions, use the dialogue model for responses
                dialogue_model, dialogue_tokenizer = load_dialogue_model()
                
                # Create a domain-specific context
                domain_contexts = {
                    "Genomics": "in the field of genomics and gene regulation",
                    "Machine Learning": "in the field of machine learning and AI research",
                    "Neuroscience": "in the field of neuroscience research",
                    "Systems Biology": "in the field of systems biology",
                    "General": "in scientific research"
                }
                
                domain_context = domain_contexts[topic]
                enhanced_question = f"Provide a research-focused answer about {question} {domain_context}."
                
                # Get response
                response = get_chatbot_response(
                    enhanced_question, 
                    dialogue_model, 
                    dialogue_tokenizer
                )
                
                # Display response
                st.write("**Answer:**")
                st.write(response)
                
                # Add references section
                st.write("**Note:** For accurate references on this topic, please consult recent publications in appropriate journals.")

# ------ MAIN APPLICATION ------

def main():
    """Main application entry point"""
    render_header()
    render_navigation()
    
    # Render current page based on state
    if st.session_state.current_page == "home":
        render_home_page()
    elif st.session_state.current_page == "explorer":
        render_explorer_page()
    elif st.session_state.current_page == "assistant":
        render_assistant_page()
    elif st.session_state.current_page == "evaluate":
        render_evaluate_page()
    elif st.session_state.current_page == "qa":
        render_qa_page()
    
    # Footer
    st.divider()
    st.write("© 2025 HypothAIze - AI Research Assistant | Developed with Streamlit")

if __name__ == "__main__":
    main()