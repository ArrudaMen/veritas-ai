import streamlit as st
import os
import datetime
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from supabase import create_client, Client

# --- 1. CONFIGURAÇÃO VISUAL E LIMPEZA ---
st.set_page_config(page_title="Veritas AI", page_icon="✝️", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    .main { background-color: #fcfcfc; }
    .welcome-container { text-align: center; padding: 40px 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CONFIGURAÇÃO DE SEGREDOS (GROQ E SUPABASE) ---
try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception as e:
    st.error("⚠️ Faltam chaves nos Secrets! Verifique o GROQ e o SUPABASE.")
    st.stop()

# Conectando com a I.A. e com o Banco de Dados Real
client_groq = Groq(api_key=GROQ_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 3. CONTROLE DE SESSÃO LOCAL ---
if "logado" not in st.session_state:
    st.session_state.logado = False
    st.session_state.usuario = ""
    st.session_state.nome_completo = ""
    st.session_state.mensagens = []

# --- 4. MENU LATERAL (LOGIN E CADASTRO REAL) ---
with st.sidebar:
    st.markdown("### 👤 Área do Usuário")
    
    if not st.session_state.logado:
        aba = st.radio("Escolha:", ["Entrar", "Criar Conta"], horizontal=True, label_visibility="collapsed")
        
        if aba == "Entrar":
            usuario_login = st.text_input("Usuário", placeholder="Ex: maria", key="login_user")
            senha_login = st.text_input("Senha", type="password", key="login_pass")
            
            if st.button("Entrar", use_container_width=True):
                try:
                    # 🔎 Vai no banco de dados procurar o usuário
                    resposta = supabase.table("usuarios").select("*").eq("usuario", usuario_login).eq("senha", senha_login).execute()
                    
                    if len(resposta.data) > 0:
                        st.session_state.logado = True
                        st.session_state.usuario = usuario_login
                        st.session_state.nome_completo = resposta.data[0]["nome_completo"]
                        
                        # 📚 Puxa o histórico de mensagens desse usuário!
                        historico = supabase.table("mensagens").select("role, content").eq("usuario", usuario_login).order("id").execute()
                        st.session_state.mensagens = historico.data
                        
                        st.rerun()
                    else:
                        st.error("Usuário ou senha inválidos.")
                except Exception as e:
                    st.error(f"Erro ao conectar com o banco: {e}")
                    
        elif aba == "Criar Conta":
            nome_cadastro = st.text_input("Nome Completo", key="cad_nome")
            usuario_cadastro = st.text_input("Nome de Usuário (Login)", key="cad_user")
            nascimento_cadastro = st.date_input("Data de Nasc.", format="DD/MM/YYYY", min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today(), value=None, key="cad_nasc")
            senha_cadastro = st.text_input("Senha", type="password", key="cad_pass")
            
            if st.button("Cadastrar", use_container_width=True):
                if not nome_cadastro or not usuario_cadastro or not senha_cadastro or not nascimento_cadastro:
                    st.warning("Preencha todos os campos obrigatórios!")
                else:
                    try:
                        # 💾 Tenta salvar no Supabase
                        supabase.table("usuarios").insert({
                            "usuario": usuario_cadastro,
                            "nome_completo": nome_cadastro,
                            "nascimento": str(nascimento_cadastro),
                            "senha": senha_cadastro
                        }).execute()
                        st.success("✅ Conta criada! Mude para a aba 'Entrar'.")
                    except Exception as e:
                        # Agora ele verifica o erro real
                        erro_str = str(e)
                        if "duplicate key" in erro_str or "23505" in erro_str:
                            st.error("❌ Esse nome de usuário já está em uso!")
                        else:
                            st.error(f"❌ Erro no banco de dados: {erro_str}")
    else:
        st.success(f"Logado como: {st.session_state.nome_completo}")
        if st.button("Sair 🚪", use_container_width=True):
            st.session_state.logado = False
            st.session_state.usuario = ""
            st.session_state.nome_completo = ""
            st.session_state.mensagens = []
            st.rerun()

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
    if not documentos_finais: return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    textos = text_splitter.split_documents(documentos_finais)
    vectorstore = Chroma.from_documents(documents=textos, embedding=embeddings, persist_directory=diretorio_banco)
    return vectorstore

base_conhecimento = inicializar_conhecimento()

# --- 6. INTERFACE DO CHAT ---
if len(st.session_state.mensagens) == 0:
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
else:
    st.markdown("<h3 style='text-align: center; color: #8B7500; font-family: serif;'>Veritas AI</h3>", unsafe_allow_html=True)

# Imprime o histórico do chat na tela
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 7. LÓGICA DE MENSAGENS E SALVAMENTO ---
if pergunta := st.chat_input("Em que posso ajudar na sua fé hoje?"):
    
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)
        
    # 💾 Se tiver logado, salva a pergunta no Supabase!
    if st.session_state.logado:
        try:
            supabase.table("mensagens").insert({"usuario": st.session_state.usuario, "role": "user", "content": pergunta}).execute()
        except Exception as e:
            pass # Ignora erro silenciosamente se falhar o salvamento

    contexto_pdf = ""
    if base_conhecimento:
        busca = base_conhecimento.similarity_search(pergunta, k=3)
        contexto_pdf = "\n".join([doc.page_content for doc in busca])

    instrucao_sistema = f"""Você é o Veritas AI, um assistente teológico católico. Responda em no máximo 2 parágrafos. Seja direto e fiel aos dogmas. Use o seguinte contexto extraído de documentos oficiais para sua resposta: {contexto_pdf}"""

    try:
        with st.chat_message("assistant"):
            chat_completion = client_groq.chat.completions.create(
                messages=[{"role": "system", "content": instrucao_sistema}, *[{"role": m["role"], "content": m["content"]} for m in st.session_state.mensagens]],
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            resposta = chat_completion.choices[0].message.content
            st.markdown(resposta)
            
            st.session_state.mensagens.append({"role": "assistant", "content": resposta})
            
            # 💾 Se tiver logado, salva a resposta da IA no Supabase!
            if st.session_state.logado:
                try:
                    supabase.table("mensagens").insert({"usuario": st.session_state.usuario, "role": "assistant", "content": resposta}).execute()
                except:
                    pass
                
    except Exception as e:
        st.error("Erro inesperado: {}".format(e))