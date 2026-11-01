import json
import time
import streamlit as st
from pathlib import Path

import config
from chatbot import SRKRChatbotEngine

# Page Configuration - Single Main Page
st.set_page_config(
    page_title="SRKR College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Dark Theme CSS (Refined Navy / Charcoal with SRKR Gold/Blue Accents)
st.markdown("""
<style>
    /* Global Background & Typography */
    .stApp {
        background-color: #0b1120 !important;
        color: #f1f5f9 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Remove default Streamlit top margin & header whitespace */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 960px !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: rgba(11, 17, 32, 0.85) !important;
        backdrop-filter: blur(8px);
    }

    /* Compact Header Bar */
    .srkr-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .header-branding {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .header-logo-badge {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: #ffffff;
        font-size: 1.6rem;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.4);
        border: 1px solid #3b82f6;
    }
    .header-title-box {
        display: flex;
        flex-direction: column;
    }
    .header-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .header-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 400;
    }
    .header-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background-color: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #10b981;
    }

    /* Welcome Section */
    .welcome-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.2rem;
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .welcome-card strong {
        color: #38bdf8;
    }

    /* Suggested Question Chips Container */
    .suggestions-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.6rem;
    }
    
    /* Custom button styling for suggestion chips */
    div[data-testid="column"] button {
        background: #0f172a !important;
        color: #e2e8f0 !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        padding: 0.6rem 0.9rem !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="column"] button:hover {
        background: #1e293b !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.15) !important;
    }

    /* Chat Messages Styling */
    .stChatMessage {
        background-color: transparent !important;
        padding: 0.6rem 0rem !important;
    }
    
    /* Assistant Chat Bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
        color: #f1f5f9 !important;
    }
    
    /* User Chat Bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a8a 100%) !important;
        border: 1px solid #2563eb !important;
        border-radius: 16px !important;
        padding: 0.9rem 1.2rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.2) !important;
        color: #ffffff !important;
    }

    /* Links inside assistant responses */
    .stChatMessage a {
        color: #38bdf8 !important;
        text-decoration: underline !important;
        font-weight: 500 !important;
    }
    .stChatMessage a:hover {
        color: #7dd3fc !important;
    }

    /* Chat Input Styling */
    div[data-testid="stChatInput"] {
        background-color: #0b1120 !important;
    }
    div[data-testid="stChatInput"] > div {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
        color: #f8fafc !important;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25) !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #f8fafc !important;
        font-size: 0.95rem !important;
    }

    /* Reset Button in Header */
    .reset-btn-col button {
        background: rgba(239, 68, 68, 0.1) !important;
        color: #f87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 10px !important;
        padding: 0.35rem 0.8rem !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .reset-btn-col button:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border-color: #ef4444 !important;
        color: #ffffff !important;
    }

    /* Footer styling */
    .srkr-footer {
        text-align: center;
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# Cache Chatbot Engine with fixed LSTM Deep Learning model (Deep Learning Lab syllabus)
@st.cache_resource(show_spinner="Initializing SRKR College Assistant (LSTM)...")
def get_chatbot_engine():
    return SRKRChatbotEngine(
        model_name="lstm",
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        similarity_threshold=config.SIMILARITY_THRESHOLD
    )

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to SRKR Engineering College (Autonomous), Bhimavaram!**\n\n"
                "I am your official digital assistant, ready to assist prospective students, current students, "
                "parents, faculty, and visitors with verified facts about SRKR.\n\n"
                "You can ask me about **admissions, programmes, departments, hostels, placements, "
                "bus transportation, library facilities, fests (SAMAGRA/JAITRA), and campus contacts**."
            )
        }
    ]

# Instantiate Chatbot Engine (automatically loads LSTM)
engine = get_chatbot_engine()

# ==========================================
# 1. COMPACT HEADER BAR
# ==========================================
header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.markdown("""
    <div class="srkr-header">
        <div class="header-branding">
            <div class="header-logo-badge">🏛️</div>
            <div class="header-title-box">
                <div class="header-title">SRKR College Assistant</div>
                <div class="header-subtitle">Sagi Rama Krishnam Raju Engineering College (Autonomous), Bhimavaram</div>
            </div>
        </div>
        <div class="header-status-pill">
            <span class="pulse-dot"></span> Online & Ready
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="reset-btn-col">', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", width="stretch"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Welcome to SRKR Engineering College (Autonomous), Bhimavaram!**\n\n"
                    "I am your official digital assistant, ready to assist prospective students, current students, "
                    "parents, faculty, and visitors with verified facts about SRKR.\n\n"
                    "You can ask me about **admissions, programmes, departments, hostels, placements, "
                    "bus transportation, library facilities, fests (SAMAGRA/JAITRA), and campus contacts**."
                )
            }
        ]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 2. SUGGESTED QUESTION CHIPS
# ==========================================
st.markdown('<div class="suggestions-label">Explore Common Topics:</div>', unsafe_allow_html=True)

sug_col1, sug_col2, sug_col3 = st.columns(3)

suggested_items = [
    ("🎓 Admissions", "How do I get admission into SRKR?"),
    ("📚 Courses Offered", "What courses and branches are available at SRKR?"),
    ("🏛️ Departments", "Tell me about academic departments and coordinators"),
    ("🏠 Hostel Facilities", "What hostel facilities and warden contacts are provided?"),
    ("💼 Placement Records", "What are the latest 2025-26 placement statistics?"),
    ("🔬 Campus Facilities", "Tell me about the central library and ECE labs")
]

clicked_query = None

for idx, (label, query_text) in enumerate(suggested_items):
    col = [sug_col1, sug_col2, sug_col3][idx % 3]
    if col.button(label, key=f"chip_{idx}", width="stretch"):
        clicked_query = query_text

# If a suggested chip was clicked, process it as a user message
if clicked_query:
    st.session_state.messages.append({"role": "user", "content": clicked_query})
    with st.spinner("Retrieving verified campus information..."):
        response_data = engine.get_response(clicked_query)
        answer_text = response_data.get("answer", config.DEFAULT_FALLBACK_RESPONSE)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text
    })
    st.rerun()

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ==========================================
# 3. CHAT CONVERSATION HISTORY
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 4. CHAT INPUT INTERACTION
# ==========================================
user_prompt = st.chat_input("Ask anything about SRKR Engineering College (e.g., courses, hostels, placements, fee)...")

if user_prompt:
    # 1. Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Compute response with typing spinner
    with st.chat_message("assistant"):
        with st.spinner("Retrieving verified campus information..."):
            response_data = engine.get_response(user_prompt)
            answer_text = response_data.get("answer", config.DEFAULT_FALLBACK_RESPONSE)
            st.markdown(answer_text)

    # 3. Save assistant message to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text
    })

# ==========================================
# 5. FOOTER
# ==========================================
st.markdown("""
<div class="srkr-footer">
    Sagi Rama Krishnam Raju Engineering College (SRKR), Bhimavaram - 534204 &bull; 
    Official Helpline: +91 9848823332 &bull; Email: info@srkrec.ac.in &bull; 
    <a href="https://www.srkrec.ac.in" target="_blank" style="color: #38bdf8; text-decoration: none;">www.srkrec.ac.in</a>
</div>
""", unsafe_allow_html=True)
