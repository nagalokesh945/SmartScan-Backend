from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

# IMPORTANT: Change 'admin' to your actual PostgreSQL password you made earlier!
DATABASE_URL = "postgresql://postgres:admin@localhost:5432/touchless_attendance_db"
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    roll_number = Column(String, unique=True)
    name = Column(String)
    face_embedding = Column(Vector(512))

Base.metadata.create_all(bind=engine)