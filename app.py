import streamlit as st

from dotenv import load_dotenv

from pypdf import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from htmlTemplates import bot_template, user_template, css

def get_pdf_text(pdf_docs):
  text = ''
  for pdf_doc in pdf_docs:
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
      text += page.extract_text()

  return text

def create_vectorstore(text_chunks):
  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  return vectorstore

def get_text_chunks(text):
  text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  text_chunks = text_splitter.split_text(text)
  return text_chunks

def get_convesation_chain(vectorstore):
  llm = ChatOpenAI()
  memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
  )
  return conversation_chain

def handle_user_input(user_question):
  response = st.session_state.conversation({'question': user_question})
  st.session_state.chat_history = response['chat_history']

  for i, message in enumerate(st.session_state.chat_history):
    if i % 2 == 0:
      st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
      st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
  load_dotenv()

  # Streamlit settings
  st.set_page_config(page_title='PDF Q&A', page_icon=':books:')

  # Custom CSS
  st.write(css, unsafe_allow_html=True)

  if 'conversation' not in st.session_state:
    st.session_state.conversation = None

  if 'chat_history' not in st.session_state:
    st.session_state.chat_history = None

  st.header('PDF Q&A :books:')
  user_question = st.text_input('Enter your question here:')
  if user_question:
    handle_user_input(user_question)

  with st.sidebar:
     st.subheader('Your documents')
     pdf_docs = st.file_uploader('Upload your PDF files here', accept_multiple_files=True)
     if st.button('Submit'):
        with st.spinner('Processing your documents...'):
          # get pdf text
          raw_text = get_pdf_text(pdf_docs)

          # get the text chunks
          text_chunks = get_text_chunks(raw_text)

          # create vectorstore
          vectorstore = create_vectorstore(text_chunks)

          # create conversation chain
          st.session_state.conversation = get_convesation_chain(vectorstore)

if __name__ == '__main__':
    main()
