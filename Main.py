import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from dotenv import load_dotenv
from langchain_chroma import Chroma
load_dotenv()
# st.set_page_config(
#     page_title="Chat with me",
#     layout="wide")

st.title("Chat with me")
history = []
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.5,max_tokens=1096,timeout=None)
system_prompt = (
    "You are an assistant of Vivek Gupta "
    "You provide answer for whatever question asked by the user about Vivek you give point to point answer whatever is asked"
    "If asked something out of the context you say please ask question on Vivek only or if you thing its the correct question then ask Vivek to add the answer of it"
    "\n\n"
    "{context}"
)
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),

    ]
)


chat_history = []
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
chain = None
database = "aboutme"
directory_for_db = os.path.join("Databases", database)
vectordb = Chroma(persist_directory=directory_for_db, embedding_function=embeddings)

def create_chain(vectordb, llm):
    # retriever = vectordb.as_retriever()
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"score_threshold": 0.9})

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the provided context.
    prompt
    You are an assistant of Vivek Gupta 
    You provide answer for whatever question asked by the user about Vivek you give point to point answer whatever is asked
    If asked something out of the context you say please ask question on Vivek only or if you thing its the correct question then ask Vivek to add the answer of it

    <context>
    {context}
    </context>

    Question: {input}""")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def create_history_chain(vectordb,model):
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"score_threshold": 0.9})
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def run_my_rag_with_history(chain,query):
    result = chain.invoke({"input": query, "chat_history": chat_history})

    chat_history.extend(
        [
            HumanMessage(content=query),
            AIMessage(content=result["answer"]),
        ]
    )
    return result
def run_my_rag(chain, query):

    result = chain.invoke({"input": query})
    return result



chain = create_chain(vectordb, llm)
# chain = create_history_chain(vectordb, model)
if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome, You can put your questions below"}]

    # Display chat messages
for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
if (len(st.session_state.messages) > 5):
    st.session_state.messages.pop(0)
input = st.chat_input()

if input:
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Please wait, Getting your answer from RAG"):
            try:

                ans = run_my_rag_with_history(chain,input)

            except Exception as e:
                st.warning("token limit have been crossed"+ e)
            response = ans['answer']
            st.write(response)

    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
