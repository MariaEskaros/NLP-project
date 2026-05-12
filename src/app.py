import streamlit as st
import time

# Configure page settings
st.set_page_config(page_title="RAG Chatbot - Milestone 3", layout="wide")
st.title("💬 Arabic RAG System Demo")
st.caption("Retrieval-Augmented Generation built on MS1 Transcripts")

# 1. Initialize session state to maintain multi-turn history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Placeholder function to simulate LangChain RAG pipeline invocation
# You will replace this logic with your actual LangChain chain calls.
def get_rag_response(user_query, history):
    # Simulate processing delay
    time.sleep(1)
    
    # Mock return payload containing response and internal retrieval logs
    return {
        "answer": f"هذه إجابة تجريبية معتمدة على السياق المسترجع لسؤالك: '{user_query}'",
        "retrieved_context": [
            {"source": "Episode_1_chunk_12", "text": "النص الأصلي المسترجع من الحلقة الأولى بالعامية المصرية... والـ code-switching هنا."},
            {"source": "Episode_3_chunk_4", "text": "جزء آخر مسترجع يحمل دلالات المعنى المطلوب."}
        ],
        "logs": {
            "model_used": "gemini-1.5-flash",
            "prompt_strategy": "System-guided Arabic Prompt",
            "out_of_domain_flag": False
        }
    }

# 2. Render previous conversation turns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If the message contains backend logs/context, display them in an expander
        if "context_logs" in message and message["context_logs"]:
            with st.expander("🔍 View Retrieved Context & System Logs"):
                st.json(message["context_logs"])

# 3. Handle user input
if prompt := st.chat_input("اكتب سؤالك هنا... (يدعم العربية والإنجليزية)"):
    # Append user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Generate chatbot response strictly grounded in context
    with st.chat_message("assistant"):
        with st.spinner("جاري استرجاع السياق وتوليد الإجابة..."):
            try:
                # Call backend function passing query and existing session history
                response_data = get_rag_response(prompt, st.session_state.messages[:-1])
                
                answer_text = response_data["answer"]
                context_logs = {
                    "retrieved_chunks": response_data["retrieved_context"],
                    "execution_logs": response_data["logs"]
                }

                # Render final text response
                st.markdown(answer_text)
                
                # Render expandable log panel to satisfy interface log display objectives
                with st.expander("🔍 View Retrieved Context & System Logs"):
                    st.json(context_logs)
                
                # Save assistant response and corresponding trace logs to session history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer_text,
                    "context_logs": context_logs
                })
                
            except Exception as e:
                # To ensure robust error handling and prevent crashes, capture API exceptions gracefully
                error_msg = f"⚠️ حدث خطأ أثناء الاتصال بالخادم: {str(e)}"
                st.error(error_msg)