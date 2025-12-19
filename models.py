"""
Database models for the RAG chatbot application.
"""
from sqlalchemy import Column, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base
import datetime as dt

from db import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(Text)
    source = Column(Text)
    chunk_text = Column(Text)
    embedding = Column(Text)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(Text, nullable=False)
    payload = Column(Text)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Text, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, index=True, nullable=False)
    role = Column(Text, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

