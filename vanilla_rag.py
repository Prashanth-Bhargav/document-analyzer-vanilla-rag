import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


st.write("Document Analyzer")
uploaded_file = st.file_uploader("Upload Document")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # embeddings = OllamaEmbeddings(model="llama3.2")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma_db", collection_name="new_collection")

    llm = Ollama(model="llama3.2")

    prompt = PromptTemplate(
        input_variables = ["context", "question"],
        template="""
    You are an intelligent document assistant. Use the provided context from the document to answer the user’s question.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer concisely, based only on the context above. If the answer is not in the context, say "The document does not contain this information."
    """
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # or "map_reduce" for large documents
        chain_type_kwargs={"prompt": prompt}
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    # Chat input
    user_query = st.text_input("Ask a question about the document:")

    if user_query:
        # Run QA chain
        answer = qa_chain.run(user_query)

        # Append messages to history
        st.session_state.history.append({"role": "user", "content": user_query})
        st.session_state.history.append({"role": "assistant", "content": answer})

    # Display chat history
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")


    # while True:
    #     query = input("Ask a question about the document (or type 'exit' to quit): ")
    #     if query.lower() == "exit":
    #         break
    #     print("Generating answer...")
    #     answer = qa_chain.run(query)
    #     print("Answer:", answer)
