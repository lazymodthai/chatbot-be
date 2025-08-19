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
คุณต้องตอบคำถามเป็นภาษาไทยเท่านั้น
จงใช้ข้อมูลที่ให้มาต่อไปนี้เพื่อตอบคำถามของผู้ใช้ หากข้อมูลไม่เกี่ยวข้องหรือไม่เพียงพอ ให้ใช้ความรู้ทั่วไปของคุณในการสนทนาอย่างเป็นมิตรตามบทบาทที่ได้รับ
อย่าสร้างคำตอบที่ไม่เกี่ยวข้องกับคำถาม
หากไม่มีข้อมูลในเอกสาร ให้ตอบว่า "ขออภัยค่ะ/ครับ ดิฉัน/ผมไม่มีข้อมูลเกี่ยวกับเรื่องนี้ในเอกสารที่ได้รับมา"
หากเป็นข้อมูลความรู้ทั่วไปให้ตอบได้เลย (อาจหาข้อมูลจาก google)
ตอบให้กระชับที่สุดเท่าที่เป็นไปได้ และจบด้วยการถามต่อว่าต้องการข้อมูลใดอีกไหม
คุณอยู่ในประเทศไทย และวันนี้คือ {now}.

ข้อมูลที่เกี่ยวข้องจากเอกสาร:
{context}

ประวัติการสนทนา:
{chat_history}

คำถามของผู้ใช้: {question}
คำตอบของคุณ (เป็นภาษาไทยเท่านั้น และเป็นรูปแบบ html ที่มีการจัดเรียงที่สวยงาม):"""

base_prompt = PromptTemplate(
    template=template, 
    input_variables=["context", "chat_history", "question", "bot_role", "now"]
)


def get_chat_history(session_id: str):
    """
    สร้าง History object โดยใช้ SQLChatMessageHistory ซึ่งจัดการ connection เอง
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=settings.DATABASE_URL,
        table_name="chat_history"
    )

def create_conversational_chain(session_id: str, now_str: str):
    """
    สร้าง Chain โดยผนวก now_str เข้าไปใน Prompt แบบไดนามิก
    """
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

# --- ฟังก์ชันอื่นๆ ไม่เปลี่ยนแปลง ---
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