import streamlit as st
import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#1. CONFIGURAÇÃO VISUAL E LIMPEZA
st.set_page_config(page_title="Veritas AI", page_icon="✝️", layout="centered")

st.markdown("""
    <style>
    /* Esconde o cabeçalho, foto do GitHub e menus padrão */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stAppHeader {display: none;}
    .main { background-color: #fcfcfc; }
    
    .welcome-container {
        text-align: center;
        padding: 40px 10px;
    }
    </style>
    """, unsafe_allow_html=True)

#2. SISTEMA DE LOGIN OPCIONAL (MENU LATERAL)
if "logado" not in st.session_state:
    st.session_state.logado = False
    st.session_state.usuario = "Visitante"

with st.sidebar:
    st.markdown("### 💾 Seu Histórico")
    
    if not st.session_state.logado:
        st.info("Faça login para salvar suas conversas e ter um histórico personalizado (Em breve!).")
        usuario_input = st.text_input("Usuário", placeholder="Ex: thiago")
        senha_input = st.text_input("Senha", type="password")
        
        if st.button("Entrar", use_container_width=True):
            if usuario_input == "thiago" and senha_input == "123":
                st.session_state.logado = True
                st.session_state.usuario = usuario_input
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")
    else:
        st.success(f"Logado como: {st.session_state.usuario.capitalize()}")
        if st.button("Sair 🚪", use_container_width=True):
            st.session_state.logado = False
            st.session_state.usuario = "Visitante"
            st.session_state.mensagens = [] # Limpa o chat ao sair
            st.rerun()

#3. SEGREDOS E CLIENTE (GROQ)
try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Chave GROQ_API_KEY não encontrada. Configure os Secrets.")
    st.stop()

client = Groq(api_key=GROQ_KEY)

#4. FUNÇÃO PARA LER OS PDFs (RAG) OTIMIZADA
@st.cache_resource 
def inicializar_conhecimento():
    diretorio_banco = "docs/chroma"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(diretorio_banco):
        return Chroma(persist_directory=diretorio_banco, embedding_function=embeddings)
    
    pasta_docs = "docs"
    documentos_finais = []
    
    if os.path.exists(pasta_docs):
        for arquivo in os.listdir(pasta_docs):
            if arquivo.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pasta_docs, arquivo))
                documentos_finais.extend(loader.load())
    
    if not documentos_finais:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    textos = text_splitter.split_documents(documentos_finais)
    
    vectorstore = Chroma.from_documents(
        documents=textos, 
        embedding=embeddings,
        persist_directory=diretorio_banco
    )
    return vectorstore

base_conhecimento = inicializar_conhecimento()

#5. INTERFACE DINÂMICA
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

pergunta = None

if len(st.session_state.mensagens) == 0:
    #TELA DE BOAS VINDAS
    nome_exibicao = st.session_state.usuario.capitalize() if st.session_state.logado else "Peregrino(a)"
    
    st.markdown(f"""
        <div class="welcome-container">
            <h1 style='font-size: 3rem; color: #8B7500; font-family: serif;'>Olá, {nome_exibicao}</h1>
            <p style='font-size: 1.2rem; color: #555;'>O que vamos pesquisar no <b>Veritas</b> hoje?</p>
        </div>
        """, unsafe_allow_html=True)
    
    #Campo de pesquisa no MEIO da tela
    pergunta = st.chat_input("Em que posso ajudar na sua fé hoje?")
    
    #Textinho de rodapé
    st.markdown("""
        <p style='text-align: center; color: #999; font-size: 0.9rem; margin-top: 20px;'>
        <i>"Conhecereis a verdade, e a verdade vos libertará." (João 8:32)</i><br>
        A I.A. consulta o Catecismo e o Direito Canônico para respostas fiéis.
        </p>
        """, unsafe_allow_html=True)

else:
    #Se já tem conversa, mostra o título pequeno e o histórico
    st.markdown("<h3 style='text-align: center; color: #8B7500; font-family: serif;'>Veritas AI</h3>", unsafe_allow_html=True)
    
    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    #Campo de pesquisa lá EMBAIXO
    pergunta = st.chat_input("Continue sua pesquisa...")

#6. LÓGICA DE PROCESSAMENTO DA I.A.
if pergunta:
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    
    if len(st.session_state.mensagens) == 1:
        st.rerun()
        
    with st.chat_message("user"):
        st.markdown(pergunta)

    #BUSCA NO PDF
    contexto_pdf = ""
    if base_conhecimento:
        busca = base_conhecimento.similarity_search(pergunta, k=3)
        contexto_pdf = "\n".join([doc.page_content for doc in busca])

    #PROMPT PARA A IA
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