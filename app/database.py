# app/database.py

from sqlalchemy import (create_engine, MetaData, Table, Column, Integer, String, Text)
from sqlalchemy.dialects.postgresql import JSONB
from .config import settings

engine = create_engine(settings.DATABASE_URL)
metadata = MetaData()

# ปรับเปลี่ยนโครงสร้างตารางให้ตรงกับที่ SQLChatMessageHistory ต้องการ
# โดยทั่วไปจะใช้ชื่อตารางว่า "message_store" เป็นค่าเริ่มต้น
message_store_table = Table(
    "message_store", # ชื่อตารางมาตรฐานของ SQLChatMessageHistory
    metadata,
    Column("id", Integer, primary_key=True),
    Column("session_id", String(255), nullable=False, index=True),
    Column("message", Text, nullable=False), # ชนิดข้อมูลเป็น Text หรือ JSONB ก็ได้
)

# เก็บตารางเดิมไว้เผื่อมีการใช้งานส่วนอื่น แต่แนะนำให้ย้ายไปใช้ message_store
chat_history_table = Table(
    "chat_history",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("session_id", String(255), nullable=False, index=True),
    Column("user_email", String(255), nullable=True),
    Column("message", Text, nullable=False),
)