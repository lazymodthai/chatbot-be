# app/main.py

import os
import shutil
import uuid
from datetime import datetime

from fastapi import (FastAPI, Query, UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .database import engine, metadata, chat_history_table
from .config import HELLO_MESSAGE, settings
from .llm_logic import (add_new_documents_to_db, add_qa_to_db, add_text_to_db,
                          create_conversational_chain, get_chat_history)

app = FastAPI(
    title="AI Chatbot Backend",
    description="Backend services for the AI Chatbot with session management and data ingestion.",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup():
    print("Application startup: Creating database tables if they don't exist...")
    metadata.create_all(engine)
    print("Database tables check complete.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str | None = Query(default=None),
    user_email: str | None = Query(default=None)
):
    await websocket.accept()

    if not user_email:
        await websocket.close(code=4001, reason="User email is required")
        return

    current_session_id = session_id
    if not current_session_id or current_session_id == "null":
        current_session_id = str(uuid.uuid4())
    
    await websocket.send_json({"type": "hello", "message": HELLO_MESSAGE, "session_id": current_session_id})

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            question = data.get("text", "")
            
            if message_type == "question":
                now_str = datetime.now().strftime('%d %B %Y')
                
                # สร้าง Chain ใหม่ทุกครั้งที่รับข้อความ เพื่อความเสถียร
                chain = await run_in_threadpool(create_conversational_chain, current_session_id, now_str)
                
                # เรียก invoke โดยส่งเฉพาะ question
                result = await run_in_threadpool(chain.invoke, {"question": question})
                answer = result["answer"]

                def update_user_email_sync():
                    try:
                        with engine.connect() as connection:
                            with connection.begin():
                                update_query = chat_history_table.update().where(
                                    chat_history_table.c.session_id == current_session_id
                                ).values(user_email=user_email)
                                connection.execute(update_query)
                    except Exception as e:
                        print(f"Could not update user_email for session {current_session_id}: {e}")
                
                await run_in_threadpool(update_user_email_sync)
                await run_in_threadpool(add_qa_to_db, question, answer)
                
                await websocket.send_json({"type": "answer", "answer": answer})

    except WebSocketDisconnect:
        print(f"Client disconnected gracefully from session {current_session_id}.")
    except Exception as e:
        print(f"An unexpected error occurred in session {current_session_id}: {e}")
        if not websocket.client_state == 'DISCONNECTED':
            await websocket.close(code=1011, reason=f"An internal error occurred.")


def _get_formatted_history_sync(session_id: str):
    if not session_id:
        return []
    history = get_chat_history(session_id)
    formatted_messages = []
    for msg in history.messages:
        sender = 'user' if msg.type == 'human' else 'ai'
        # content ของ SQLChatMessageHistory เป็น string ที่ต้อง decode
        # แต่ในกรณีนี้ langchain อาจจัดการให้แล้ว ถ้าไม่ ให้ใช้ json.loads(msg.content)
        formatted_messages.append({"sender": sender, "text": msg.content})
    return formatted_messages

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    return await run_in_threadpool(_get_formatted_history_sync, session_id)


@app.get("/sessions")
async def get_sessions(user_email: str):
    def _get_sessions_sync():
        if not user_email: return []
        query = text("""
            SELECT session_id FROM (
                SELECT session_id, MAX(id) as max_id
                FROM chat_history WHERE user_email = :user_email
                GROUP BY session_id
            ) as latest_sessions ORDER BY max_id DESC;
        """)
        try:
            with engine.connect() as connection:
                result = connection.execute(query, {"user_email": user_email})
                return [row[0] for row in result]
        except Exception as e:
            print(f"Could not fetch sessions: {e}")
            return []
    return await run_in_threadpool(_get_sessions_sync)

# --- Endpoint ที่เหลือไม่มีการเปลี่ยนแปลง ---
@app.post("/uploadfile/")
async def upload_file(file: UploadFile):
    def _save_file_and_learn():
        data_path = "data"
        os.makedirs(data_path, exist_ok=True)
        file_location = os.path.join(data_path, file.filename)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        add_new_documents_to_db()
        return {"info": f"File '{file.filename}' processed and learned."}
    return await run_in_threadpool(_save_file_and_learn)

@app.post("/add-text/")
async def add_text(data: dict):
    text = data.get("text")
    if not text:
        return {"error": "No text provided."}
    await run_in_threadpool(add_text_to_db, text)
    return {"info": "Text processed and learned."}