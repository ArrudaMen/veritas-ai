import streamlit as st
import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. CONFIGURAÇÃO VISUAL E LIMPEZA ---
st.set_page_config(page_title="Veritas AI", page_icon="✝️", layout="centered")

st.markdown("""
    <style>
    /* Esconde o cabeçalho, foto do GitHub e menus */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stAppHeader {display: none;}

    /* Cor de fundo e estilos */
    .main { background-color: #fcfcfc; }
    
    /* Centraliza as boas-vindas */
    .welcome-container {
        text-align: center;
        padding: 40px 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SEGREDOS E CLIENTE (GROQ) ---
try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Chave GROQ_API_KEY não encontrada.")
    st.stop()

client = Groq(api_key=GROQ_KEY)

# --- 3. FUNÇÃO PARA LER OS PDFs (RAG) OTIMIZADA ---
@st.cache_resource 
def inicializar_conhecimento():
    diretorio_banco = "docs/chroma"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Se o banco de dados já estiver salvo no PC/Servidor, apenas carrega (MUITO RÁPIDO)
    if os.path.exists(diretorio_banco):
        return Chroma(persist_directory=diretorio_banco, embedding_function=embeddings)
    
    # Se não existir, lê os PDFs e cria o banco (Só roda 1 vez)
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
    
    # Cria o banco e SALVA no disco (Isso acaba com o carregamento infinito)
    vectorstore = Chroma.from_documents(
        documents=textos, 
        embedding=embeddings,
        persist_directory=diretorio_banco
    )
    return vectorstore

# Inicializa o banco de dados
base_conhecimento = inicializar_conhecimento()

# --- 4. INTERFACE DINÂMICA (BOAS-VINDAS VS CHAT) ---
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Se NÃO houver mensagens, mostra a explicação centralizada
if len(st.session_state.mensagens) == 0:
    st.markdown("""
        <div class="welcome-container">
            <h1 style='font-size: 3rem; color: #8B7500; font-family: serif;'>Veritas AI</h1>
            <p style='font-size: 1.2rem; color: #555;'><b>Veritas</b> vem do latim e significa <b>Verdade</b>.</p>
            <p style='margin-bottom: 20px;'>
                Esta I.A. consulta o Catecismo, o Direito Canônico e as Escrituras 
                para trazer respostas fiéis à tradição católica.
            </p>
            <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px; border-left: 5px solid #8B7500; font-style: italic;'>
                "Conhecereis a verdade, e a verdade vos libertará." (João 8:32)
            </div>
            <p style='margin-top: 30px; color: #999; font-size: 0.9rem;'>👇 Pergunte algo no campo abaixo para começar</p>
        </div>
        """, unsafe_allow_html=True)
else:
    # Se já tem conversa, mostra só o título pequeno no topo
    st.markdown("<h3 style='text-align: center; color: #8B7500; font-family: serif;'>Veritas AI</h3>", unsafe_allow_html=True)

# Imprime o histórico de mensagens na tela
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 5. CAMPO DE CHAT E LÓGICA DA I.A. ---
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