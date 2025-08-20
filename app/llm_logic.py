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
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers import ContextualCompressionRetriever
from typing import List

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

template = """คุณคือ AI ผู้ช่วยที่ต้องตอบคำถามจาก "ข้อมูลที่เกี่ยวข้องจากเอกสาร" ที่ให้มาเท่านั้น
บทบาทของคุณคือ: '{bot_role}'

**กฏเหล็กที่ต้องปฏิบัติตามอย่างเคร่งครัด:**
1. ตอบคำถามของผู้ใช้โดยอ้างอิงจาก "ข้อมูลที่เกี่ยวข้องจากเอกสาร" และบทบาทของคุณ **เท่านั้น**
2. **ห้ามใช้ความรู้ส่วนตัวหรือข้อมูลภายนอกที่ไม่ได้มาจากเอกสารโดยเด็ดขาด**
3. หาก "ข้อมูลที่เกี่ยวข้องจากเอกสาร" ไม่มีเนื้อหา, ว่างเปล่า, หรือไม่เกี่ยวข้องกับคำถาม ให้ตอบว่า: **"เรายังไม่มีข้อมูลในส่วนนี้ค่ะ"** ห้ามพยายามตอบหรือเดาคำตอบเด็ดขาด
4. ตอบเป็นภาษาไทยเท่านั้นและจัดรูปแบบด้วย GitHub Flavored Markdown หากจำเป็น
5. ห้ามใช้แท็ก HTML, ไม่ต้องทวนคำถาม, และไม่ต้องกล่าวทักทาย
6. จะต้องตอบลงท้ายประโยคด้วย "ค่ะ" เท่านั้น
ึ7. หากคำถามบอกว่าข้อมูลที่ให้นั้นผิด ให้หาคำตอบใหม่ไม่ให้เหมือนคำตอบก่อนหน้า
8. หากไม่จำเป็น คำตอบไม่ต้องมีหัวข้อ

คำแนะนำในการจัดรูปแบบ:
- ใช้หัวข้อ (เช่น `# หัวข้อหลัก`, `## หัวข้อย่อย`)
- ใช้รายการ (Bulleted lists) โดยใช้เครื่องหมาย `*` หรือ `-`
- ใช้ตัวหนา (`**ข้อความ**`) เพื่อเน้นคำสำคัญ
- ใช้ Horizontal Rule (`---`) เพื่อคั่นส่วนที่ไม่เกี่ยวข้องกัน

คุณอยู่ในประเทศไทย และวันนี้คือ {now}. ให้ใช้รูปแบบคำตอบเป็นปี ค.ศ.

---
**ข้อมูลที่เกี่ยวข้องจากเอกสาร:**
{context}
---

**ประวัติการสนทนา:**
{chat_history}

**คำถามของผู้ใช้:** {question}
**คำตอบของคุณ (ตอบจาก "ข้อมูลที่เกี่ยวข้องจากเอกสาร" เท่านั้น):**"""


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

class RelevanceScorer(BaseDocumentCompressor):
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks = None,
    ) -> List[Document]:
        """
        ให้คะแนนและจัดลำดับเอกสารใหม่ตามกฎที่กำหนด
        """
        scored_docs = []
        for doc in documents:
            score = 0
            if doc.metadata.get("source") == "user_correction":
                score += 100
            elif doc.metadata.get("source") == "qa_learning":
                score += 50

            if doc.metadata.get("importance") == "high":
                score += 80

            doc.metadata["relevance_score"] = score
            scored_docs.append(doc)

        sorted_docs = sorted(scored_docs, key=lambda x: x.metadata["relevance_score"], reverse=True)
        return sorted_docs

def create_conversational_chain(session_id: str, now_str: str, scope: dict | None = None):
    dynamic_prompt = base_prompt.partial(bot_role=BOT_ROLE, now=now_str)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=get_chat_history(session_id),
        return_messages=True
    )

    search_kwargs = {}
    if scope:
        search_kwargs['filter'] = scope

    base_retriever = vector_store.as_retriever(search_kwargs={'filter': scope} if scope else {})

    scorer = RelevanceScorer()

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=scorer,
        base_retriever=base_retriever
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=compression_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": dynamic_prompt},
    )
    return chain

def add_new_documents_to_db():
    docs = load_and_split_documents()
    if docs:
        vector_store.add_documents(docs)
        print(f"Added {len(docs)} new document chunks to the database.")

def add_text_to_db(text: str, source: str, category: str):
    docs = split_text_into_docs(text)
    for doc in docs:
        doc.metadata["source"] = source
        doc.metadata["category"] = category
    vector_store.add_documents(docs)
    print(f"Added new text from '{source}' in category '{category}' to the database.")

def add_qa_to_db(question: str, answer: str):
    qa_text = f"คำถามที่เคยมีผู้ถาม: {question}\nคำตอบที่ถูกต้อง: {answer}"
    doc = Document(
        page_content=qa_text,
        metadata={"source": "qa_learning", "category": "faq"}
    )
    vector_store.add_documents([doc])
    print(f"Learned new Q&A: {question}")

def add_correction_to_db(question: str, corrected_answer: str):
    correction_text = f"ข้อมูลที่ได้รับการแก้ไขโดยผู้ใช้: เมื่อถูกถามว่า '{question}', คำตอบที่ถูกต้องคือ '{corrected_answer}'"
    doc = Document(
        page_content=correction_text,
        metadata={"source": "user_correction", "category": "correction", "importance": "high"} # เพิ่มระดับความสำคัญ
    )
    vector_store.add_documents([doc])
    print(f"Learned new correction for question: {question}")