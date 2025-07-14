from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import re
import unicodedata
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
    
load_dotenv()

from sklearn.feature_extraction.text import TfidfVectorizer

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please add it to your .env file.")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,  # Slightly increased for less repetitive responses
    openai_api_key=OPENAI_API_KEY,
)

# Load vector store
vs = FAISS.load_local("faiss_index_langchain", embeddings, allow_dangerous_deserialization=True)

# IMPROVED: Better retrieval configuration
retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5, "score_threshold": 0.5}  # Reduced k, lowered threshold
)

# IMPROVED: Simplified and clearer contextualization prompt
contextualise_prompt = """
Given the chat history and the latest user question, rephrase the question to be standalone and clear.

If the question is already clear and standalone, return it unchanged.
If the question refers to previous context (like "that", "this", "it"), make it specific.

Chat History: {chat_history}
Current Question: {input}

Reformulated Question:"""

contextualise_prompt_template = ChatPromptTemplate.from_messages([
    ("system", contextualise_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualise_prompt_template,
)

# IMPROVED: Cleaner, more focused system prompt
system_prompt = """
You are a scientific paper analyst. Answer the user's question based on the provided context from research papers.

Instructions:
1. Provide a direct, specific answer to the question
2. Quote relevant passages from the papers when appropriate
3. Cite the source papers/sections
4. If the context doesn't contain enough information, say so clearly
5. Be concise but comprehensive

Context: {context}
Question: {input}

Answer:"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain,
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    words = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)

def keyword_extraction(text):
    """Extract keywords using TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    return list(feature_names)

# IMPROVED: Simplified response quality check
def is_response_inadequate(response_text):
    """Check if response is inadequate using simple heuristics"""
    text = response_text.lower().strip()
    
    # Check for typical inadequate response patterns
    inadequate_patterns = [
        r"i (can't|cannot|am unable to|do not|don't) (find|help|provide|have)",
        r"please (specify|provide more details|clarify)",
        r"i don't have enough information",
        r"i cannot provide",
        r"unfortunately, i cannot",
        r"i'm sorry, but i cannot"
    ]
    
    # Check if response is too short (likely inadequate)
    if len(text.split()) < 15:
        return True
        
    # Check for inadequate patterns
    for pattern in inadequate_patterns:
        if re.search(pattern, text):
            return True
    
    return False

def get_answer(query, session_id="default", max_retries=1):
    """Main function to get answers from the RAG system"""
    try:
        # Get response from conversational RAG chain
        response = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        
        answer = response.get("answer", "")
        
        # IMPROVED: Simplified inadequate response handling
        if is_response_inadequate(answer) and max_retries > 0:
            # Try with a more specific prompt
            enhanced_query = f"Based on the research papers, provide specific details about: {query}"
            try:
                response = conversational_rag_chain.invoke(
                    {"input": enhanced_query},
                    config={"configurable": {"session_id": session_id}}
                )
                new_answer = response.get("answer", answer)
                # Only use new answer if it's better
                if not is_response_inadequate(new_answer):
                    answer = new_answer
            except Exception:
                pass  # Keep original answer if enhancement fails
        
        return {
            "answer": answer,
            "source_documents": response.get("context", []),
            "keywords": keyword_extraction(query)
        }
        
    except Exception as e:
        return {
            "answer": f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question.",
            "source_documents": [],
            "keywords": []
        }

async def get_answer_async(query, session_id="default", max_retries=1):
    """Async version of get_answer"""
    import asyncio
    
    try:
        # Get response from conversational RAG chain
        response = await asyncio.to_thread(
            conversational_rag_chain.invoke,
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        
        answer = response.get("answer", "")
        
        # Check for inadequate response
        if is_response_inadequate(answer) and max_retries > 0:
            enhanced_query = f"Based on the research papers, provide specific details about: {query}"
            try:
                response = await asyncio.to_thread(
                    conversational_rag_chain.invoke,
                    {"input": enhanced_query},
                    config={"configurable": {"session_id": session_id}}
                )
                new_answer = response.get("answer", answer)
                if not is_response_inadequate(new_answer):
                    answer = new_answer
            except Exception:
                pass
        
        return {
            "answer": answer,
            "source_documents": response.get("context", []),
            "keywords": await asyncio.to_thread(keyword_extraction, query)
        }
        
    except Exception as e:
        return {
            "answer": f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question.",
            "source_documents": [],
            "keywords": []
        }

# Example usage
if __name__ == "__main__":
    import uuid
    print("Welcome to the Improved Scientific Paper Q&A Chat!")
    print("Type your question and press Enter. Type 'exit' to quit.\n")
    
    session_id = str(uuid.uuid4())
    
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        result = get_answer(user_input, session_id=session_id)
        print(f"\nAnswer: {result['answer']}")
        
        # Optional: Show source documents
        if result['source_documents']:
            print(f"\nSources: {len(result['source_documents'])} documents retrieved")
        
        print("-" * 50)