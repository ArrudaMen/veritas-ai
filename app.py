import streamlit as st
import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. CONFIGURAÇÃO VISUAL E LIMPEZA ---
st.set_page_config(page_title="Veritas AI", page_icon="✝️", layout="centered")

# CONSERTO DO BUG DO MENU LATERIAL: Tirei o "header hidden"
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;} /* Esconde só o botão de deploy lá em cima */
    .main { background-color: #fcfcfc; }
    
    .welcome-container {
        text-align: center;
        padding: 40px 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. BANCO DE DADOS TEMPORÁRIO E SESSÃO ---
# Simulando um banco de dados para testar a criação de contas
if "banco_usuarios" not in st.session_state:
    st.session_state.banco_usuarios = {
        "admin": {"senha": "123", "nome": "Administrador", "nascimento": "01/01/1990"}
    }

if "logado" not in st.session_state:
    st.session_state.logado = False
    st.session_state.usuario = ""
    st.session_state.nome_completo = ""

# --- 3. MENU LATERAL (LOGIN E CADASTRO) ---
with st.sidebar:
    st.markdown("### 👤 Área do Usuário")
    
    if not st.session_state.logado:
        # Alterna entre Login e Criar Conta
        aba = st.radio("Escolha uma opção:", ["Entrar", "Criar Conta"], horizontal=True, label_visibility="collapsed")
        
        if aba == "Entrar":
            usuario_login = st.text_input("Usuário", placeholder="Ex: maria")
            senha_login = st.text_input("Senha", type="password")
            
            if st.button("Entrar", use_container_width=True):
                # Verifica se o usuário existe no "banco" e se a senha bate
                if usuario_login in st.session_state.banco_usuarios and st.session_state.banco_usuarios[usuario_login]["senha"] == senha_login:
                    st.session_state.logado = True
                    st.session_state.usuario = usuario_login
                    st.session_state.nome_completo = st.session_state.banco_usuarios[usuario_login]["nome"]
                    st.rerun()
                else:
                    st.error("Usuário ou senha inválidos.")
                    
        elif aba == "Criar Conta":
            nome_cadastro = st.text_input("Nome Completo")
            usuario_cadastro = st.text_input("Nome de Usuário (Login)")
            nascimento_cadastro = st.date_input("Data de Nascimento", format="DD/MM/YYYY")
            senha_cadastro = st.text_input("Senha", type="password")
            
            if st.button("Cadastrar", use_container_width=True):
                if not nome_cadastro or not usuario_cadastro or not senha_cadastro:
                    st.warning("Preencha todos os campos obrigatórios!")
                elif usuario_cadastro in st.session_state.banco_usuarios:
                    st.error("❌ Esse nome de usuário já está em uso! Escolha outro.")
                else:
                    # Salva no banco de dados temporário
                    st.session_state.banco_usuarios[usuario_cadastro] = {
                        "senha": senha_cadastro,
                        "nome": nome_cadastro,
                        "nascimento": nascimento_cadastro
                    }
                    st.success("✅ Conta criada com sucesso! Mude para a opção 'Entrar'.")
    else:
        st.success(f"Logado como: {st.session_state.nome_completo}")
        if st.button("Sair 🚪", use_container_width=True):
            st.session_state.logado = False
            st.session_state.usuario = ""
            st.session_state.nome_completo = ""
            st.session_state.mensagens = []
            st.rerun()

# --- 4. SEGREDOS E CLIENTE (GROQ) ---
try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Chave GROQ_API_KEY não encontrada.")
    st.stop()

client = Groq(api_key=GROQ_KEY)

# --- 5. FUNÇÃO PARA LER OS PDFs (RAG) ---
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
    
    vectorstore = Chroma.from_documents(documents=textos, embedding=embeddings, persist_directory=diretorio_banco)
    return vectorstore

base_conhecimento = inicializar_conhecimento()

# --- 6. INTERFACE DINÂMICA (ESTILO GEMINI) ---
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
    
if "primeira_pergunta" not in st.session_state:
    st.session_state.primeira_pergunta = None

# TELA INICIAL LIMPA (SE NÃO TIVER CONVERSA)
if len(st.session_state.mensagens) == 0 and st.session_state.primeira_pergunta is None:
    
    st.markdown("""
        <div class="welcome-container">
            <h1 style='font-size: 3rem; color: #8B7500; font-family: serif;'>Veritas AI</h1>
            <p style='font-size: 1.2rem; color: #555; margin-bottom: 30px;'>
                <b>Veritas</b> vem do latim e significa <b>Verdade</b>.
            </p>
            <p style='margin-bottom: 20px;'>
                Esta I.A. consulta o Catecismo, o Direito Canônico e as Escrituras 
                para trazer respostas fiéis à tradição católica.
            </p>
            <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px; border-left: 5px solid #8B7500; font-style: italic; margin-bottom: 30px; display: inline-block;'>
                "Conhecereis a verdade, e a verdade vos libertará." (João 8:32)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        pergunta_centro = st.text_input("Pesquisa", placeholder="Em que posso ajudar na sua fé hoje?", label_visibility="collapsed")
        if pergunta_centro:
            st.session_state.primeira_pergunta = pergunta_centro
            st.rerun()

else:
    # VISUAL DE CHAT
    st.markdown("<h3 style='text-align: center; color: #8B7500; font-family: serif;'>Veritas AI</h3>", unsafe_allow_html=True)
    
    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    pergunta_baixo = st.chat_input("Continue sua pesquisa...")
    
    pergunta_submetida = None
    if st.session_state.primeira_pergunta:
        pergunta_submetida = st.session_state.primeira_pergunta
        st.session_state.primeira_pergunta = None
    elif pergunta_baixo:
        pergunta_submetida = pergunta_baixo
        
    # PROCESSAMENTO
    if pergunta_submetida:
        st.session_state.mensagens.append({"role": "user", "content": pergunta_submetida})
        
        with st.chat_message("user"):
            st.markdown(pergunta_submetida)

        contexto_pdf = ""
        if base_conhecimento:
            busca = base_conhecimento.similarity_search(pergunta_submetida, k=3)
            contexto_pdf = "\n".join([doc.page_content for doc in busca])

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