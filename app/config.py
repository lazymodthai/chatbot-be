from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "chatbot_db")

settings = Settings()

def load_text_from_file(filepath, default_text):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return default_text

BOT_ROLE = load_text_from_file("role.txt", "คุณคือผู้ช่วย AI ที่เป็นมิตรและช่วยเหลือผู้อื่น")
HELLO_MESSAGE = load_text_from_file("hello.txt", "สวัสดีครับ มีอะไรให้ช่วยไหม?")