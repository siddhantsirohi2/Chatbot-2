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
retriever = vs.as_retriever(search_type = "similarity",search_kwargs={"k": 8, "score_threshold": 0.7})

contextualise_prompt = """
You are a question reformulator for a scientific paper Q&A system.

**Your task**: Analyze the current question and chat history to determine if the question needs reformulation.

**Rules**:
1. If the current question is clear, specific, and standalone → return it AS-IS
2. If the question references previous context (e.g., "What about that?", "How does this relate?") → reformulate to be standalone
3. If the question is ambiguous → make it more specific
4. NEVER add information not implied by the question

**Examples**:
- "What is machine learning?" → "What is machine learning?" (clear, return as-is)
- "How does that work?" → reformulate based on context
- "Tell me more about the results" → reformulate based on context

**Current question**: {input}
**Chat history**: {chat_history}

Return ONLY the reformulated question, nothing else.
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
You are an expert scientific paper analyst. Your primary goal is to answer the CURRENT question accurately and specifically.

**CRITICAL**: Focus on the CURRENT question first. Only use chat history for context if the current question explicitly references it.

**Answer Structure**:
1. **Direct Answer**: Answer the current question first, clearly and specifically
2. **Supporting Evidence**: Cite specific papers/sections that support your answer
3. **Context**: Only mention previous conversation if the current question references it

**Guidelines**:
- Be specific and direct
- Cite exact paper sections when possible
- Distinguish between facts and interpretations
- If information is limited, say so clearly
- Prioritize accuracy over completeness

**Current Question**: {input}
**Available Context**: {context}

Answer the current question using the provided context. Be specific and cite sources.
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
        r"specify the actual issue",
        r"i don't have enough information",
        r"i cannot provide",
        r"unfortunately, i cannot",
        r"i'm sorry, but i cannot"
    ]
    return any(re.search(pat, text) for pat in vague_patterns)

def is_vague_response_llm(input_string: str) -> bool:
    prompt = f"""Analyze this answer for vagueness or lack of specific information.

Answer: "{input_string}"

Consider:
1. Does it provide specific, actionable information?
2. Does it cite specific sources or papers?
3. Does it avoid generic responses like "I cannot help" or "Please specify"?
4. Does it give a direct answer to the question?

Reply only with 'Yes' (if vague) or 'No' (if specific and helpful).
"""
    try:
        response = llm.invoke(prompt)
        return response.content.strip().lower() == "yes"
    except Exception:
        # Fallback to regex check if LLM fails
        return is_vague_response_regex(input_string)

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
            # Try to get a more specific answer with different approach
            refined_query = f"Provide a detailed, specific answer with citations to: {query}"
            try:
                response = conversational_rag_chain.invoke(
                    {"input": refined_query},
                    config={"configurable": {"session_id": session_id}}
                )
                new_answer = response.get("answer", answer)
                # Only use new answer if it's not also vague
                if not is_vague_response_regex(new_answer):
                    answer = new_answer
            except Exception:
                pass  # Keep original answer if refinement fails
        
        return {
            "answer": answer,
            "source_documents": response.get("context", []),
            "keywords": keyword_extraction(query)
        }
        
    except Exception as e:
        return {
            "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or check if the backend services are running properly.",
            "source_documents": [],
            "keywords": []
        }

# Example usage
if __name__ == "__main__":
    import uuid
    print("Welcome to the CLI Scientific Paper Q&A Chat!")
    print("Type your question and press Enter. Type 'exit' to quit.\n")
    session_id = str(uuid.uuid4())
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        # Get the reformulated question from the contextualizer
        chat_history = get_session_history(session_id).messages
        from langchain_core.messages import HumanMessage, AIMessage
        formatted_history = []
        for msg in chat_history:
            if msg.type == 'human':
                formatted_history.append(HumanMessage(content=msg.content))
            elif msg.type == 'ai':
                formatted_history.append(AIMessage(content=msg.content))
        # Use the contextualise_prompt_template and llm to get the reformulated question
        prompt = contextualise_prompt_template.format(input=user_input, chat_history=formatted_history)
        reformulated = llm.invoke(prompt).content
        print("\n--- Reformulated Question (Contextualizer Output) ---")
        print(reformulated)
        # Get the full RAG answer
        result = get_answer(user_input, session_id=session_id)
        print("\n--- System Prompt Answer (RAG Output) ---")
        print(result["answer"])
        print("")
