from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ConversationBufferMemory
import shutil
import os
from datetime import datetime
import uuid

from .config import HELLO_MESSAGE
from .llm_logic import create_conversational_chain, add_new_documents_to_db, add_text_to_db, add_qa_to_db

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connection_memories = {}

class ConnectionManager:
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        connection_memories[websocket] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        await websocket.send_json({"type": "hello", "message": HELLO_MESSAGE})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str | None = Header(default=None)):
    await websocket.accept()
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    chain = create_conversational_chain(session_id)
    await websocket.send_json({"type": "hello", "message": HELLO_MESSAGE, "session_id": session_id})

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            question = data.get("text", "")
            
            if message_type == "question":
                current_time = datetime.now().strftime('%d %B %Y เวลา %H:%M')
                enhanced_question = f"{question}\n\n[วันเวลาปัจจุบัน: {current_time}]"
                
                result = chain.invoke({"question": enhanced_question})
                answer = result["answer"]
                
                add_qa_to_db(question, answer)
                
                await websocket.send_json({"type": "answer", "answer": answer})

    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected.")
    except Exception as e:
        print(f"Error for session {session_id}: {e}")

@app.post("/uploadfile/")
async def upload_file(file: UploadFile):
    file_location = os.path.join("data", file.filename)
    os.makedirs("data", exist_ok=True)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    add_new_documents_to_db()
    return {"info": "File processed and learned."}

@app.post("/add-text/")
async def add_text(data: dict):
    text = data.get("text")
    if text:
        add_text_to_db(text)
        return {"info": "Text processed and learned."}
    return {"error": "No text provided."}