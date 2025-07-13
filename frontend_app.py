import streamlit as st
import requests
from datetime import datetime

API_URL = "http://localhost:8000"  # Adjust if backend runs elsewhere

st.set_page_config(page_title="Scientific Paper Q&A", layout="centered")

if "jwt" not in st.session_state:
    st.session_state.jwt = None
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Auth & Session ---
with st.sidebar:
    st.title("🔐 Session")
    if st.session_state.jwt:
        st.success(f"Logged in as: {st.session_state.username}")
        if st.button("Log out"):
            st.session_state.jwt = None
            st.session_state.username = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        auth_tab = st.radio("Account", ["Log In", "Sign Up"])
        if auth_tab == "Log In":
            with st.form("login_form"):
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                submit = st.form_submit_button("Log In")
                if submit:
                    resp = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
                    if resp.status_code == 200:
                        token = resp.json()["access_token"]
                        st.session_state.jwt = token
                        st.session_state.username = username
                        st.success("Logged in!")
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "Login failed."))
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

# --- Main Chat Interface ---
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
                data = {"query": user_input, "session_id": st.session_state.username}
                with st.spinner("AI is thinking..."):
                    resp = requests.post(f"{API_URL}/ask", json=data, headers=headers)
                if resp.status_code == 200:
                    answer = resp.json()["answer"]
                    st.session_state.chat_history.append((user_input, answer))
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Error getting answer."))
else:
    st.warning("Please log in or sign up from the sidebar to start chatting.") 