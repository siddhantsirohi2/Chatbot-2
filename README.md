# Scientific Paper Q&A Application

This project is an end-to-end Question & Answer platform for scientific papers, powered by Retrieval-Augmented Generation (RAG) and modern AI. It features a FastAPI backend, a Streamlit frontend, user authentication, and persistent chat history.

## Features
- **User Signup/Login** with JWT authentication
- **Ask questions** about scientific papers and get AI-powered answers
- **Chat history** with session management
- **Frontend**: Streamlit app for interactive chat
- **Backend**: FastAPI for API endpoints and authentication
- **Database**: (Current: SQLite, Easy to extend to MongoDB)
- **CORS** enabled for frontend-backend communication

---

## Project Structure
```
generative ai 2/
├── backend/
│   ├── app22.py           # (AI answer logic)
│   ├── back.py            # FastAPI backend (main API)
│   └── ...
├── frontend_app.py        # Streamlit frontend
└── README.md              # This file
```

---

## Setup Instructions

### 1. Clone the repository
```sh
git clone <your-repo-url>
cd generative\ ai\ 2
```

### 2. Backend Setup (FastAPI)
- Create a virtual environment and activate it:
  ```sh
  cd backend
  python -m venv venv
  venv\Scripts\activate  # On Windows
  # or
  source venv/bin/activate  # On Mac/Linux
  ```
- Install dependencies:
  ```sh
  pip install fastapi uvicorn passlib[bcrypt] python-jose sqlalchemy
  ```
- Run the backend:
  ```sh
  uvicorn back:app --reload
  ```
  The API will be available at `http://localhost:8000`

### 3. Frontend Setup (Streamlit)
- Install dependencies:
  ```sh
  pip install streamlit requests
  ```
- Run the frontend:
  ```sh
  streamlit run frontend_app.py
  ```
  The app will open in your browser (default: `http://localhost:8501`)

---

## Authentication
- User passwords are securely hashed (bcrypt)
- JWT tokens are used for authentication and stored in Streamlit session state
- All protected endpoints require a valid JWT in the `Authorization` header

---

## Database
- **Default:** SQLite (via SQLAlchemy, see `backend/back.py`)
- **Extensible:** Schema and code can be adapted for MongoDB or other databases
- User and chat data are stored persistently

---

## API Endpoints (Backend)
- `POST /signup` — Register a new user
- `POST /login` — Authenticate and get JWT token
- `POST /ask` — Ask a question (requires JWT)

---

## Contributing
1. Fork the repo and create your branch
2. Make your changes and commit
3. Push to your fork and submit a pull request

---

## License
MIT License

---

## Acknowledgements
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Passlib](https://passlib.readthedocs.io/) 