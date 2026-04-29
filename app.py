import streamlit as st
import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURAÇÃO VISUAL ---
st.set_page_config(page_title="Veritas AI", page_icon="📜", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    h1 { color: #8B7500; text-align: center; font-family: 'serif'; }
    .intro-text { text-align: center; color: #555; font-size: 1.1rem; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

# --- SEGREDOS E CLIENTE ---
try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Chave GROQ_API_KEY não encontrada.")
    st.stop()

client = Groq(api_key=GROQ_KEY)

# --- FUNÇÃO PARA LER OS PDFs (RAG) ---
@st.cache_resource # Isso evita que ele releia os PDFs toda vez que o site atualizar
def inicializar_conhecimento():
    pasta_docs = "docs"
    documentos_finais = []
    
    if os.path.exists(pasta_docs):
        for arquivo in os.listdir(pasta_docs):
            if arquivo.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pasta_docs, arquivo))
                documentos_finais.extend(loader.load())
    
    if not documentos_finais:
        return None

    # Quebra o texto em pedaços para a IA não se perder
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    textos = text_splitter.split_documents(documentos_finais)
    
    # Cria o "mapa" de conhecimento usando o seu processador (Embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=textos, embedding=embeddings)
    return vectorstore

# Inicializa o banco de dados de PDFs
base_conhecimento = inicializar_conhecimento()

# --- INTERFACE ---
st.markdown("<h1>Veritas AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='intro-text'>Focada no ensino do catolicismo, baseada na fé, na tradição e nas escrituras sagradas.</div>", unsafe_allow_html=True)

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if pergunta := st.chat_input("Em que posso ajudar na sua fé hoje?"):
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # BUSCA NO PDF
    contexto_pdf = ""
    if base_conhecimento:
        busca = base_conhecimento.similarity_search(pergunta, k=3)
        contexto_pdf = "\n".join([doc.page_content for doc in busca])

    # PROMPT PARA A IA
    instrucao_sistema = f"""
    Você é o Veritas AI, um assistente teológico católico.
    Responda em no máximo 2 parágrafos. Seja direto e fiel aos dogmas.
    Use o seguinte contexto extraído de documentos oficiais para sua resposta:
    {contexto_pdf}
    """

    try:
        with st.chat_message("assistant"):
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": instrucao_sistema},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.mensagens]
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            resposta = chat_completion.choices[0].message.content
            st.markdown(resposta)
            st.session_state.mensagens.append({"role": "assistant", "content": resposta})
    except Exception as e:
        st.error("Erro inesperado: {}".format(e))