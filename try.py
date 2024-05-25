# import dotenv
# import streamlit as st
# import fitz  # PyMuPDF
# from langchain import hub
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.llms import HuggingFaceHub
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_community.vectorstores import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# dotenv.load_dotenv()

# # Define a class to encapsulate document sections
# class Document:
#     def __init__(self, content, metadata=None):
#         self.page_content = content
#         if metadata is None:
#             self.metadata = {}  # Default metadata as an empty dictionary
#         else:
#             self.metadata = metadata

# # Streamlit app interface
# st.set_page_config(page_title="FestBot: Upload fest doc to solve queries", page_icon="", layout='centered')
# st.markdown("<h3 style='background:#0284fe;padding:20px;border-radius:10px;text-align:center;'>Chat with PDF document</h3>",
#             unsafe_allow_html=True)
# st.markdown("")
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
# if uploaded_file is not None:
#     doc = fitz.open(stream=uploaded_file.read())
#     # Extract text from the entire PDF
#     # text = ""
#     # for page in doc:
#     #     text += page.get_text()
#     # loader = PyMuPDFLoader(uploaded_file.name)
#     # doc = loader.load()
#     embeddings = HuggingFaceEmbeddings()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     text = text_splitter.split_documents(doc)
#     db = Chroma.from_documents(text, embeddings)
#     llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-1-pythia-12b")
#     chain = load_qa_chain(llm, chain_type="stuff")
#     # Initialize model
#     # llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-1-pythia-12b")

#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#     # React to user input
#     if prompt := st.chat_input("You can ask me questions about the document."):
#         # Display user message in chat message container
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         # Generate response
#         response = llm.invoke([("human", prompt)])
#         # Access response metadata
#         metadata = response.response_metadata
#         # Display assistant response in chat message container
#         with st.chat_message("assistant"):
#             st.markdown(str(response))
#             st.markdown(f"Response Metadata: {metadata}")
#         # Add assistant response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": response})

#     if st.button("Clear chat"):
#         st.session_state.clear()

import os
import dotenv
import streamlit as st
import fitz  # PyMuPDF
from langchain import hub
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
dotenv.load_dotenv()

# Set HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Define a class to encapsulate document sections
class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        if metadata is None:
            self.metadata = {}  # Default metadata as an empty dictionary
        else:
            self.metadata = metadata

# Streamlit app interface
st.set_page_config(page_title="FestBot: Upload fest doc to solve queries", page_icon="", layout='centered')
st.markdown("<h3 style='background:#0284fe;padding:20px;border-radius:10px;text-align:center;'>Chat with PDF document</h3>",
            unsafe_allow_html=True)
st.markdown("")

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    # Open the PDF with PyMuPDF directly from the uploaded file
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    # Extract text from the entire PDF
    text = ""
    for page in doc:
        text += page.get_text()

    # Initialize embeddings and text splitter
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(text)

    # Create document objects
    documents = [Document(content=chunk) for chunk in text_chunks]

    # Initialize vector store and LLM
    db = Chroma.from_documents(documents, embeddings)
    llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-1-pythia-12b")
    chain = load_qa_chain(llm, chain_type="stuff")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("You can ask me questions about the document."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        response = chain({"question": prompt})
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    if st.button("Clear chat"):
        st.session_state.clear()



# import dotenv
# import streamlit as st
# import fitz  # PyMuPDF
# from openai import OpenAI

# dotenv.load_dotenv()

# # Point to the local server
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# # Define a class to encapsulate document sections
# class Document:
#     def __init__(self, content, metadata=None):
#         self.page_content = content
#         if metadata is None:
#             self.metadata = {}  # Default metadata as an empty dictionary
#         else:
#             self.metadata = metadata

# # Streamlit app interface
# st.set_page_config(page_title="Chat with PDF", page_icon="", layout='centered')
# st.markdown("<h3 style='background:#0284fe;padding:20px;border-radius:10px;text-align:center;'>Chat with PDF document</h3>",
#             unsafe_allow_html=True)
# st.markdown("")
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
# if uploaded_file is not None:
#     doc = fitz.open(stream=uploaded_file.read())

#     # Extract text from the entire PDF
#     text = ""
#     for page in doc:
#         text += page.get_text()

#     # Text splitting
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.split_text(text)

#     # Wrap splits into Document objects
#     documents = [Document(split) for split in splits]

#     # Initialize and populate the vectorstore
#     vectorstore = Chroma.from_documents(documents=documents,
#                                          embedding=OpenAIEmbeddings(openai_api_key="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"))

#     # Retrieval and generation
#     retriever = vectorstore.as_retriever()
#     prompt = hub.pull("rlm/rag-prompt")
#     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)

#     rag_chain = (
#             {"context": retriever | format_docs, "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#     )

#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # React to user input
#     if prompt := st.chat_input("You can ask me questions about the document."):
#         # Display user message in chat message container
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         # Display assistant response in chat message container
#         with st.chat_message("assistant"):
#             completion = client.chat.completions.create(
#                 model="local-model",  # this field is currently unused
#                 messages=[
#                     {"role": "system", "content": "Answer in sarcasm"},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.7,
#             )
#             st.markdown(completion.choices[0].message.content)
#         # Add assistant response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": completion.choices[0].message.content})

#     if st.button("Clear chat"):
#         st.session_state.clear()

