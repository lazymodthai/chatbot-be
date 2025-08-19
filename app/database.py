from sqlalchemy import create_engine
from .config import settings

engine = create_engine(settings.DATABASE_URL)

def get_db_connection():
    return engine.connect()

# อาจเพิ่ม setup table ต่างๆ ที่นี่ถ้าจำเป็น