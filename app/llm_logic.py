# app/llm_logic.py

import locale
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.vectorstores.pgvector import PGVector
from sqlalchemy.engine.base import Connection

from .config import settings, BOT_ROLE
from .services import load_and_split_documents, split_text_into_docs

try:
    locale.setlocale(locale.LC_TIME, 'th_TH.UTF-8')
except locale.Error:
    print("Locale 'th_TH.UTF-8' not supported, using system default.")

embeddings_model = OllamaEmbeddings(model="llama3")
llm_model = Ollama(model="llama3")

vector_store = PGVector(
    connection_string=settings.DATABASE_URL,
    embedding_function=embeddings_model,
    collection_name=settings.COLLECTION_NAME,
)

template = """คุณคือ AI ผู้ช่วยที่ต้องสวมบทบาทต่อไปนี้: '{bot_role}'
คุณต้องตอบคำถามเป็นภาษาไทยเท่านั้น และต้องจัดรูปแบบคำตอบทั้งหมดโดยใช้ GitHub Flavored Markdown เสมอ
ห้ามใช้แท็ก HTML ในคำตอบของคุณโดยเด็ดขาด ไม่ต้องทวนคำถาม ไม่ต้องสวัสดีทุกรอบ

คำแนะนำในการจัดรูปแบบ:
- ใช้หัวข้อ (เช่น `# หัวข้อหลัก`, `## หัวข้อย่อย`) เพื่อแบ่งส่วนคำตอบให้ชัดเจน
- ใช้รายการ (Bulleted lists) โดยใช้เครื่องหมาย `*` หรือ `-` สำหรับข้อมูลที่เป็นข้อๆ
- ใช้ตัวหนา (`**ข้อความ**`) เพื่อเน้นคำสำคัญหรือใจความหลัก
- ใช้ลิงก์ (`[ข้อความลิงก์](URL)`) เมื่อมีการอ้างอิงถึงเว็บไซต์
- ใช้ Horizontal Rule (`---`) เพื่อคั่นส่วนที่ไม่เกี่ยวข้องกัน

คุณอยู่ในประเทศไทย และวันนี้คือ {now}.

ข้อมูลที่เกี่ยวข้องจากเอกสาร:
{context}

ประวัติการสนทนา:
{chat_history}

คำถามของผู้ใช้: {question}
คำตอบของคุณ (เป็นภาษาไทย, จัดรูปแบบด้วย Markdown):"""

base_prompt = PromptTemplate(
    template=template, 
    input_variables=["context", "chat_history", "question", "bot_role", "now"]
)

def get_chat_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=settings.DATABASE_URL,
        table_name="chat_history"
    )

def create_conversational_chain(session_id: str, now_str: str):
    dynamic_prompt = base_prompt.partial(bot_role=BOT_ROLE, now=now_str)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=get_chat_history(session_id),
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": dynamic_prompt},
    )
    return chain

def add_new_documents_to_db():
    docs = load_and_split_documents()
    if docs:
        vector_store.add_documents(docs)
        print(f"Added {len(docs)} new document chunks to the database.")

def add_text_to_db(text: str):
    docs = split_text_into_docs(text)
    vector_store.add_documents(docs)
    print("Added new text to the database.")

def add_qa_to_db(question: str, answer: str):
    qa_text = f"คำถามที่เคยมีผู้ถาม: {question}\nคำตอบที่ถูกต้อง: {answer}"
    doc = Document(page_content=qa_text, metadata={"source": "qa_learning"})
    vector_store.add_documents([doc])
    print(f"Learned new Q&A: {question}")

def add_correction_to_db(question: str, corrected_answer: str):
    """
    บันทึกคำตอบที่ผู้ใช้แก้ไขแล้วลงใน Vector Store
    เพื่อให้ AI ใช้เป็นข้อมูลอ้างอิงที่มีความสำคัญสูงในอนาคต
    """
    correction_text = f"ข้อมูลที่ได้รับการแก้ไขโดยผู้ใช้: เมื่อถูกถามว่า '{question}', คำตอบที่ถูกต้องคือ '{corrected_answer}'"
    doc = Document(page_content=correction_text, metadata={"source": "user_correction"})
    vector_store.add_documents([doc])
    print(f"Learned new correction for question: {question}")