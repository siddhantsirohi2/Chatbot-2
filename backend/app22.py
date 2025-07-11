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
# Handle stopwords import

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
    
load_dotenv()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Correct environment variable name for OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please add it to your .env file.")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY,
)

# WARNING: allow_dangerous_deserialization=True is only safe if you trust the source of the FAISS index files.
vs = FAISS.load_local("faiss_index_langchain", embeddings, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_type = "similarity",search_kwargs={"k": 15})

contextualise_prompt = """
Given a chat history and the latest user question which might reference context in the chat history,
formulate a standalone question which can be understood without the chat history.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""

contextualise_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", contextualise_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualise_prompt_template,
)

system_prompt = """
You are an expert scientific paper analyst. Help users understand and analyze scientific literature with precision and clarity.

**When answering questions:**
- **Answer directly** first, then provide supporting context
- **Cite specific sections** from the retrieved papers
- **Explain methodology** and how findings were obtained
- **Note limitations** and potential biases
- **Use clear language** while maintaining scientific accuracy

**For different question types:**
- **Concepts**: Define clearly with examples and broader connections
- **Methods**: Explain design, procedures, and evaluate strengths/weaknesses  
- **Results**: Present key findings with statistical context
- **Interpretation**: Discuss significance, implications, and alternative explanations
- **Critical analysis**: Assess study quality, reliability, and reproducibility

Always distinguish between established facts, author claims, and your interpretations. Acknowledge uncertainty when evidence is limited.

**Context:** {context}
Use the retrieved document chunks to provide accurate, specific information from the scientific papers.
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

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

def clean_list_of_strings(strings):
    return [clean_text(s) for s in strings]

def check_english(llm_answer):
    prompt = f"""Decide whether the text is in English or not. Reply with only "English" or "Non-English"

Text:"{llm_answer}"
"""
    ans = llm.invoke(prompt)
    return ans.content

def convert_to_english(llm_answer_non_english):
    prompt = f"""Translate the below text to English

Text:"{llm_answer_non_english}"
"""
    ans = llm.invoke(prompt)
    return ans.content

def is_vague_response_regex(input_string: str) -> bool:
    text = input_string.lower().strip()
    vague_patterns = [
        r"i (can't|cannot|am unable to|do not|don't) (find|help|provide|have)",
        r"please (specify|provide more details)",
        r"how can i (help|assist)",
        r"could you please",
        r"can you please",
        r"it[- ]?related issues?",
        r"i'?m not able",
        r"specify the actual issue"
    ]
    return any(re.search(pat, text) for pat in vague_patterns)

def is_vague_response_llm(input_string: str) -> bool:
    prompt = f"""Decide whether the following answer is 'vague, unclear, or lacking sufficient information' or 'it is a reply to someone saying Thanks or appreciation like You're welcome! If you have any other questions or need further assistance, feel free to ask'. Reply only with 'Yes' or 'No'.

answer: "{input_string}"
"""
    response = llm.invoke(prompt)
    return response.content.strip().lower() == "yes"

def keyword_extraction(text):
    """Extract keywords using TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    return list(feature_names)

def get_answer(query, session_id="default"):
    """Main function to get answers from the RAG system"""
    try:
        # Get response from conversational RAG chain
        response = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        
        answer = response.get("answer", "")
        
        # Check if response is in English
        language_check = check_english(answer)
        if language_check.strip() != "English":
            answer = convert_to_english(answer)
        
        # Check if response is vague
        is_vague_regex = is_vague_response_regex(answer)
        is_vague_llm = is_vague_response_llm(answer)
        
        if is_vague_regex or is_vague_llm:
            # Try to get a more specific answer
            refined_query = f"Please provide a detailed and specific answer to: {query}"
            response = conversational_rag_chain.invoke(
                {"input": refined_query},
                config={"configurable": {"session_id": session_id}}
            )
            answer = response.get("answer", answer)
        
        return {
            "answer": answer,
            "source_documents": response.get("context", []),
            "keywords": keyword_extraction(query)
        }
        
    except Exception as e:
        return {
            "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
            "source_documents": [],
            "keywords": []
        }

# Example usage
if __name__ == "__main__":
    # Test the system
    test_query = "What are the applications of machine learning in healthcare?"
    result = get_answer(test_query)
    
    print("Query:", test_query)
    print("Answer:", result["answer"])
    print("Keywords:", result["keywords"])
    print("Number of source documents:", len(result["source_documents"]))
