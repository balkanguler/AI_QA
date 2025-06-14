import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

class RAGPipelineGemini:
    def __init__(self):
        self.embedding_model = None
        self.qa_data = []
        self.embeddings = None
        self.index = None
        self.gemini_model = None
        
    @st.cache_resource
    def load_models(_self):
        """Load embedding model and initialize Gemini"""
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return embedding_model
    
    def initialize_gemini(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """Initialize Gemini API"""
        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(model_name)
            return True
        except Exception as e:
            st.error(f"Error initializing Gemini: {str(e)}")
            return False
    
    def test_gemini_connection(self):
        """Test if Gemini API is working"""
        try:
            response = self.gemini_model.generate_content("Hello, respond with 'API working'")
            return "API working" in response.text
        except Exception as e:
            st.error(f"Gemini API test failed: {str(e)}")
            return False
    
    def load_qa_data(self, file_path: str):
        """Load questions and answers from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both single object and array formats
            if isinstance(data, list):
                self.qa_data = data
            else:
                self.qa_data = [data]
                
            st.success(f"Loaded {len(self.qa_data)} Q&A pairs")
            return True
        except Exception as e:
            st.error(f"Error loading QA data: {str(e)}")
            return False
    
    def create_embeddings(self):
        """Create embeddings for all questions"""
        if not self.qa_data:
            st.error("No QA data loaded")
            return False
            
        if self.embedding_model is None:
            self.embedding_model = self.load_models()
        
        questions = [item['question'] for item in self.qa_data]
        
        with st.spinner("Creating embeddings..."):
            self.embeddings = self.embedding_model.encode(questions)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        st.success("Embeddings created and indexed successfully!")
        return True
    
    def find_similar_questions(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        """Find k most similar questions to the query"""
        if self.index is None or self.embedding_model is None:
            return []
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar questions
        scores, indices = self.index.search(query_embedding, min(k, len(self.qa_data)))
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.qa_data):
                results.append((self.qa_data[idx], float(score)))
        
        return results
    
    def generate_answer(self, query: str, similar_qas: List[Tuple[Dict, float]]) -> str:
        """Generate answer using Gemini with context from similar Q&As"""
        if not similar_qas:
            return "No similar questions found in the database."
        
        if self.gemini_model is None:
            return "Gemini API not initialized. Please check your API key."
        
        # Prepare context from similar Q&As
        context = "Based on the following questions and answers from the knowledge base:\n\n"
        for i, (qa, score) in enumerate(similar_qas, 1):
            context += f"{i}. Question: {qa['question']}\n   Answer: {qa['answer']}\n"
            if qa.get('answer_url'):
                context += f"   Source: {qa['answer_url']}\n"
            context += "\n"
        
        # Create prompt for Gemini
        prompt = f"""{context}

User Question: {query}

IMPORTANT INSTRUCTIONS:
1. Answer the user's question ONLY if the answer can be found in the provided questions and answers above.
2. If the answer is not available in the provided knowledge base, respond with "I cannot answer this question based on the available information in the knowledge base."
3. If you can answer based on the knowledge base, provide a clear and helpful response.
4. Do not use information outside of the provided context.
5. If relevant, mention which question(s) from the knowledge base your answer is based on.
6. Answer with the same language as the user's question.

Answer:"""
        
        try:
            # Generate response using Gemini
            with st.spinner("Generating answer with Gemini..."):
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=512,
                        top_p=0.9,
                        top_k=40
                    )
                )
            
            # Extract the generated text
            answer = response.text.strip()
            
            return answer if answer else "I cannot generate an answer based on the provided information."
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"

def main():
    st.set_page_config(
        page_title="RAG Pipeline with Gemini",
        page_icon="💎",
        layout="wide"
    )
    
    st.title("💎 RAG Pipeline with Gemini API")
    st.markdown("Upload your Q&A data and ask questions to get relevant answers powered by Google Gemini!")
    
    # Initialize RAG pipeline
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipelineGemini()
    
    rag = st.session_state.rag_pipeline
    
    # Sidebar for API key and setup
    with st.sidebar:
        st.header("🔧 Setup")
        
        # Gemini API Configuration
        st.subheader("💎 Gemini API Settings")
        
        # API Key input
        api_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey",
            value=st.session_state.get('gemini_api_key', '')
        )
        
        if api_key:
            st.session_state.gemini_api_key = api_key
        
        # Model selection
        model_options = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.0-pro"
        ]
        
        selected_model = st.selectbox(
            "Select Gemini Model:",
            options=model_options,
            index=0,
            help="Flash: Fastest, Pro: Most capable, 1.0: Legacy"
        )
        
        # Initialize Gemini button
        if api_key and st.button("🚀 Initialize Gemini API"):
            if rag.initialize_gemini(api_key, selected_model):
                if rag.test_gemini_connection():
                    st.success("✅ Gemini API connected successfully!")
                    st.session_state.gemini_ready = True
                else:
                    st.error("❌ Gemini API test failed")
            else:
                st.error("❌ Failed to initialize Gemini API")
        
        # Show API key instructions
        if not api_key:
            with st.expander("📋 How to get Gemini API Key"):
                st.markdown("""
                1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
                2. Sign in with your Google account
                3. Click "Create API Key"
                4. Copy the API key and paste it above
                5. Click "Initialize Gemini API"
                
                **Note**: Gemini API has generous free usage limits!
                """)
        
        st.divider()
        
        # File upload section
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload QAS JSON file",
            type=['json'],
            help="Upload your questions and answers JSON file"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with open("temp_qas.json", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load QA data
            if st.button("📥 Load Q&A Data"):
                if rag.load_qa_data("temp_qas.json"):
                    st.session_state.data_loaded = True
        
        # Create embeddings button
        if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
            if st.button("🔗 Create Embeddings"):
                if rag.create_embeddings():
                    st.session_state.embeddings_ready = True
        
        st.divider()
        
        # Display stats
        st.subheader("📊 Status")
        
        # API Status
        if hasattr(st.session_state, 'gemini_ready') and st.session_state.gemini_ready:
            st.success(f"✅ Gemini API ({selected_model})")
        else:
            st.error("❌ Gemini API")
        
        # Data Status
        if rag.qa_data:
            st.metric("Q&A Pairs Loaded", len(rag.qa_data))
        else:
            st.metric("Q&A Pairs Loaded", 0)
        
        # Embeddings Status
        if hasattr(st.session_state, 'embeddings_ready') and st.session_state.embeddings_ready:
            st.success("✅ Embeddings Ready")
        else:
            st.error("❌ Embeddings Not Ready")
        
        # Overall Status
        if (hasattr(st.session_state, 'gemini_ready') and st.session_state.gemini_ready and
            hasattr(st.session_state, 'embeddings_ready') and st.session_state.embeddings_ready):
            st.success("🎉 System Ready!")
    
    # Main interface
    if (hasattr(st.session_state, 'gemini_ready') and st.session_state.gemini_ready and
        hasattr(st.session_state, 'embeddings_ready') and st.session_state.embeddings_ready):
        
        st.header("💬 Ask a Question")
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            placeholder="Type your question here...",
            height=100,
            key="query_input"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            num_similar = st.slider(
                "Number of similar questions to retrieve:",
                min_value=3,
                max_value=15,
                value=10,
                help="More questions provide more context but may include noise"
            )
            
            show_similarity_scores = st.checkbox(
                "Show similarity scores",
                value=True,
                help="Display how similar each question is to your query"
            )
        
        col1, col2 = st.columns([1, 1])
        
        if st.button("🔍 Search & Answer", type="primary", use_container_width=True) and query:
            
            with col1:
                st.subheader("📋 Similar Questions Found")
                
                # Find similar questions
                similar_qas = rag.find_similar_questions(query, k=num_similar)
                
                if similar_qas:
                    for i, (qa, score) in enumerate(similar_qas, 1):
                        with st.expander(
                            f"#{i} - {qa['question'][:60]}..." if len(qa['question']) > 60 
                            else f"#{i} - {qa['question']}"
                        ):
                            st.write(f"**Question:** {qa['question']}")
                            st.write(f"**Answer:** {qa['answer']}")
                            if qa.get('answer_url'):
                                st.write(f"**Source:** [{qa['answer_url']}]({qa['answer_url']})")
                            if show_similarity_scores:
                                st.write(f"**Similarity Score:** {score:.3f}")
                else:
                    st.warning("No similar questions found.")
            
            with col2:
                st.subheader("🤖 Gemini Generated Answer")
                
                if similar_qas:
                    # Generate answer using Gemini
                    answer = rag.generate_answer(query, similar_qas)
                    
                    # Display answer in a nice format
                    st.markdown("### Answer:")
                    st.write(answer)
                    
                    # Show confidence metrics
                    st.markdown("### Confidence Metrics:")
                    avg_similarity = np.mean([score for _, score in similar_qas[:3]])
                    col2a, col2b = st.columns(2)
                    
                    with col2a:
                        st.metric("Avg Similarity (Top 3)", f"{avg_similarity:.3f}")
                    with col2b:
                        st.metric("Sources Found", len(similar_qas))
                    
                    # Confidence interpretation
                    if avg_similarity > 0.8:
                        st.success("🟢 High confidence - Very similar questions found")
                    elif avg_similarity > 0.6:
                        st.warning("🟡 Medium confidence - Somewhat similar questions found")
                    else:
                        st.error("🔴 Low confidence - No very similar questions found")
                    
                else:
                    st.error("Cannot generate answer without similar questions.")
    
    else:
        # Setup instructions
        st.info("👆 Please complete the setup steps in the sidebar to get started.")
        
        # Show example format
        st.subheader("📝 Expected JSON Format")
        st.code('''[
  {
    "question": "What is machine learning?",
    "answer": "Machine learning is a subset of AI that enables computers to learn and make decisions from data without being explicitly programmed.",
    "answer_url": "https://example.com/ml-guide"
  },
  {
    "question": "How does deep learning work?",
    "answer": "Deep learning uses neural networks with multiple layers to process and learn from data, automatically extracting features and patterns.",
    "answer_url": "https://example.com/dl-guide"
  },
  {
    "question": "What are the types of machine learning?",
    "answer": "The main types are supervised learning, unsupervised learning, and reinforcement learning.",
    "answer_url": "https://example.com/ml-types"
  }
]''', language='json')
        
        # Benefits of using Gemini
        st.subheader("✨ Why Gemini API?")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🚀 Easy Setup**
            - No local installation
            - Just need API key
            - Works immediately
            """)
        
        with col2:
            st.markdown("""
            **⚡ High Performance**
            - Fast response times
            - High-quality answers
            - Latest AI technology
            """)
        
        with col3:
            st.markdown("""
            **💰 Cost Effective**
            - Generous free tier
            - Pay per use
            - No hardware needed
            """)

if __name__ == "__main__":
    main()