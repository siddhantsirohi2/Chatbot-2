import streamlit as st
import requests
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Scientific Paper Q&A",
    page_icon="🧑‍🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- API & SESSION STATE ---
API_URL = "http://localhost:8000"

# Initialize session state variables
if "jwt" not in st.session_state:
    st.session_state.jwt = None
if "username" not in st.session_state:
    st.session_state.username = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR: AUTH & INFO ---
with st.sidebar:
    st.title("🔐 Session")
    
    if st.session_state.jwt:
        st.success(f"Logged in as: {st.session_state.username}")
        st.info(f"Session ID: {st.session_state.session_id[:8]}...")
        if st.button("Log out"):
            st.session_state.jwt = None
            st.session_state.username = None
            st.session_state.session_id = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        auth_tab = st.radio("Account", ["Log In", "Sign Up"])
        
        # --- Login Form ---
        if auth_tab == "Log In":
            with st.form("login_form"):
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                submit = st.form_submit_button("Log In")
                if submit:
                    resp = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
                    if resp.status_code == 200:
                        token_data = resp.json()
                        st.session_state.jwt = token_data["access_token"]
                        st.session_state.username = username
                        st.session_state.session_id = token_data["session_id"]
                        st.success("Logged in!")
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "Login failed."))

        # --- Signup Form ---
        else:
            with st.form("signup_form"):
                username = st.text_input("Username", key="signup_user")
                password = st.text_input("Password", type="password", key="signup_pass")
                submit = st.form_submit_button("Sign Up")
                if submit:
                    resp = requests.post(f"{API_URL}/signup", json={"username": username, "password": password})
                    if resp.status_code == 200:
                        st.success("Signup successful! Please log in.")
                    else:
                        st.error(resp.json().get("detail", "Signup failed."))

    st.markdown("---")
    st.info("This is a RAG-powered scientific paper Q&A chat. Ask anything about your research topic!")

# --- MAIN CHAT INTERFACE ---
st.title("🧑‍🔬 Scientific Paper Q&A Chat")

if st.session_state.jwt:
    chat_placeholder = st.container()
    with chat_placeholder:
        # Display chat history as bubbles
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div style='background-color:#1565c0; color:white; border-radius:10px; padding:10px; margin-bottom:5px; width:fit-content; max-width:80%; margin-left:auto; box-shadow: 0 2px 8px rgba(21,101,192,0.08);'>
                <b>You:</b> {q}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background-color:#f9f9f9; color:#222; border-radius:10px; padding:10px; margin-bottom:15px; width:fit-content; max-width:80%; box-shadow: 0 2px 8px rgba(0,0,0,0.04);'>
                <b>AI:</b> {a}
            </div>
            """, unsafe_allow_html=True)

        # Chat input at the bottom
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("Type your question and press Send", key="chat_input", height=50)
            send = st.form_submit_button("Send")
            if send and user_input.strip():
                headers = {"Authorization": f"Bearer {st.session_state.jwt}"}
                data = {"query": user_input, "session_id": st.session_state.session_id}
                with st.spinner("AI is thinking..."):
                    try:
                        resp = requests.post(f"{API_URL}/ask", json=data, headers=headers)
                        if resp.status_code == 200:
                            answer = resp.json()["answer"]
                            st.session_state.chat_history.append((user_input, answer))
                            st.rerun()
                        else:
                            error_detail = resp.json().get("detail", "Unknown error")
                            st.error(f"Error: {error_detail}")
                            # Add error to chat history for debugging
                            st.session_state.chat_history.append((user_input, f"Error: {error_detail}"))
                            st.rerun()
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend. Make sure the backend is running on http://localhost:8000")
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
else:
    st.warning("Please log in or sign up from the sidebar to start chatting.")