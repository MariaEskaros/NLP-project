import sys
from pathlib import Path
import csv 
from datetime import datetime 
import streamlit as st


# =========================
# Project setup
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))


# =========================
# Project imports
# =========================

from embeddings import EmbeddingModel
from vector_store import ChromaVectorStore
from retriever import Retriever
from conversation_memory import ConversationMemory
from semantic_cache import SemanticCache
from rag_pipeline_ import RAGPipeline

from llm_client_groq import GroqLLM
from llm_client_openai import OpenAILLM
from llm_client_hf import HuggingFaceLLM


# =========================
# Streamlit config
# =========================

st.set_page_config(
    page_title="Arabic RAG Chatbot - Milestone 3",
    layout="wide"
)


# =========================
# Load RAG components
# =========================

@st.cache_resource
def load_rag_components():
    embedding_model = EmbeddingModel()

    vector_store = ChromaVectorStore(
        persist_dir=PROJECT_ROOT / "data" / "vector_db",
        collection_name="arabic_transcripts_ms3"
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=5
    )

    semantic_cache = SemanticCache(
        embedding_model=embedding_model,
        cache_path=PROJECT_ROOT / "data" / "cache" / "semantic_cache.json",
        similarity_threshold=0.85
    )

    return embedding_model, vector_store, retriever, semantic_cache


embedding_model, vector_store, retriever, semantic_cache = load_rag_components()


# =========================
# Session state
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()


# =========================
# LLM factory
# =========================

def create_llm(provider_name: str, model_name: str):
    if provider_name == "Groq":
        return GroqLLM(model_name=model_name)

    # if provider_name == "OpenAI":
    #     return OpenAILLM(model_name=model_name)

    if provider_name == "HuggingFace":
        return HuggingFaceLLM(model_name=model_name)

    raise ValueError(f"Unknown provider: {provider_name}")


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.header("⚙️ إعدادات النظام")

    provider_name = st.selectbox(
        "اختر مزود النموذج:",
        ["Groq", "HuggingFace"]
    )

    model_options = {
        "Groq": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile"
        ],
        # "OpenAI": [
        #     "gpt-3.5-turbo",
        #     "gpt-4o-mini"
        # ],
        "HuggingFace": [
            "Qwen/Qwen2.5-7B-Instruct"
        ]
    }

    selected_model = st.selectbox(
        "اختر النموذج:",
        model_options[provider_name]
    )

    memory_strategy = st.selectbox(
        "Memory strategy:",
        ["sliding", "full", "truncated", "summary"]
    )

    use_cache = st.checkbox(
        "Use semantic cache",
        value=True
    )

    out_of_domain_threshold = st.slider(
        "Out-of-domain threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.2,
        step=0.05
    )

    if st.button("Clear chat memory"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.success("Chat memory cleared.")

    if st.button("Clear semantic cache"):
        semantic_cache.clear()
        st.success("Semantic cache cleared.")

    st.caption(f"Provider: {provider_name}")
    st.caption(f"Model: {selected_model}")


# =========================
# RAG response function
# =========================

def get_rag_response(user_query: str, provider_name: str, model_name: str):
    llm = create_llm(
        provider_name=provider_name,
        model_name=model_name
    )

    rag = RAGPipeline(
        retriever=retriever,
        llm=llm,
        memory=st.session_state.memory,
        memory_strategy=memory_strategy,
        last_n_turns=2,
        semantic_cache=semantic_cache,
        use_cache=use_cache,
        out_of_domain_threshold=out_of_domain_threshold,
        max_retries=2
    )

    result = rag.answer(user_query)

    retrieved_context = []

    for chunk in result.get("retrieved_chunks", []):
        retrieved_context.append({
            "rank": chunk.get("rank"),
            "episode": chunk.get("episode"),
            "source_file": chunk.get("source_file"),
            "distance": chunk.get("distance"),
            "relevance_score": chunk.get("relevance_score"),
            "text_preview": chunk.get("text", "")[:1000]
        })

    return {
        "answer": result.get("answer"),
        "retrieved_context": retrieved_context,
        "logs": {
            "provider": provider_name,
            "model_used": result.get("model"),
            "cache_hit": result.get("cache_hit"),
            "matched_question": result.get("matched_question"),
            "cache_similarity": result.get("cache_similarity"),
            "out_of_domain": result.get("out_of_domain"),
            "retrieval_score": result.get("retrieval_score"),
            "memory_strategy": result.get("memory_strategy"),
            "prompt_chars": result.get("prompt_chars"),
            "error": result.get("error")
        }
    }

# =========================
# Evaluation Logger
# =========================
def log_interaction(query, response_data):
    # Ensure the logs directory exists inside your data folder
    log_dir = PROJECT_ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "evaluation_logs.csv"
    file_exists = log_file.exists()
    
    # Figure out what to write for the context status
    logs = response_data["logs"]
    chunks = response_data["retrieved_context"]
    
    if logs.get("cache_hit"):
        context_status = "Cache Hit (0 retrieved)"
    else:
        context_status = f"Found {len(chunks)} chunks"

    # Open the CSV and append the new row (utf-8-sig ensures Arabic text displays correctly in Excel)
    with open(log_file, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        
        # Write the headers if it's a brand new file
        if not file_exists:
            writer.writerow([
                "Timestamp", "User Query", "Model Used", "Context Status", 
                "Out of Domain?", "System Response"
            ])
            
        # Write the actual data
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query,
            logs.get("model_used", "Unknown"),
            context_status,
            logs.get("out_of_domain", False),
            response_data["answer"],
            "", # Leave blank for you to fill in Excel later
            ""  # Leave blank for you to fill in Excel later
        ])
# =========================
# Main UI
# =========================

st.title("💬 Arabic RAG System Demo")
st.caption("Retrieval-Augmented Generation built on Milestone 1 transcripts")


# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("context_logs"):
            with st.expander("🔍 View Retrieved Context & System Logs"):
                st.json(message["context_logs"])


# Chat input
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("جاري استرجاع السياق وتوليد الإجابة..."):
            try:
                response_data = get_rag_response(
                    user_query=prompt,
                    provider_name=provider_name,
                    model_name=selected_model
                )

                answer_text = response_data["answer"]

                context_logs = {
                    "retrieved_chunks": response_data["retrieved_context"],
                    "execution_logs": response_data["logs"]
                }

                st.markdown(answer_text)

                with st.expander("🔍 View Retrieved Context & System Logs"):
                    st.json(context_logs)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "context_logs": context_logs
                      
                })
                log_interaction(query=prompt, response_data=response_data)  

            except Exception as e:
                error_message = f"⚠️ حدث خطأ أثناء توليد الإجابة: {str(e)}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "context_logs": None
                })