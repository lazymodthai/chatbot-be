# app/database.py

from sqlalchemy import (create_engine, MetaData, Table, Column, Integer, String, Text)
from sqlalchemy.dialects.postgresql import JSONB
from .config import settings

engine = create_engine(settings.DATABASE_URL)
metadata = MetaData()

message_store_table = Table(
    "message_store",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("session_id", String(255), nullable=False, index=True),
    Column("message", Text, nullable=False),
)

chat_history_table = Table(
    "chat_history",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("session_id", String(255), nullable=False, index=True),
    Column("user_email", String(255), nullable=True),
    Column("message", Text, nullable=False),
)