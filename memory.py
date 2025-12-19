"""
Memory service for managing conversation context and history.
Provides clean abstraction for conversation management.

This module handles:
- Conversation session management
- Message storage and retrieval
- Context building for RAG queries
- Conversation history management
"""
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from models import Conversation, ConversationMessage


class MemoryService:
    """Service for managing conversation memory and context."""

    def __init__(self, db: Session):
        self.db = db

    def get_or_create_conversation(self, session_id: str) -> Conversation:
        """Get existing conversation or create a new one."""
        try:
            conversation = self.db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if not conversation:
                conversation = Conversation(session_id=session_id)
                self.db.add(conversation)
                self.db.commit()
                self.db.refresh(conversation)
            
            return conversation
        except SQLAlchemyError as e:
            self.db.rollback()
            raise RuntimeError(f"Failed to get or create conversation: {str(e)}") from e

    def add_message(
        self, 
        conversation_id: int, 
        role: str, 
        content: str
    ) -> ConversationMessage:
        """Add a message to the conversation."""
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role}. Must be 'user' or 'assistant'")
        
        try:
            message = ConversationMessage(
                conversation_id=conversation_id,
                role=role,
                content=content
            )
            self.db.add(message)
            
            # Update conversation timestamp
            conversation = self.db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            if conversation:
                conversation.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(message)
            return message
        except SQLAlchemyError as e:
            self.db.rollback()
            raise RuntimeError(f"Failed to add message: {str(e)}") from e

    def get_conversation_history(
        self, 
        conversation_id: int, 
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Get conversation history as list of message dicts."""
        query = self.db.query(ConversationMessage).filter(
            ConversationMessage.conversation_id == conversation_id
        ).order_by(ConversationMessage.created_at)
        
        if limit:
            query = query.limit(limit)
        
        messages = query.all()
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    def get_recent_context(
        self, 
        conversation_id: int, 
        max_messages: int = 10
    ) -> str:
        """Get recent conversation context as formatted string."""
        history = self.get_conversation_history(conversation_id, limit=max_messages)
        if not history:
            return ""
        
        context_parts = []
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role_label}: {msg['content']}")
        
        return "\n\n".join(context_parts)

    def clear_conversation(self, conversation_id: int) -> bool:
        """Clear all messages from a conversation."""
        deleted = self.db.query(ConversationMessage).filter(
            ConversationMessage.conversation_id == conversation_id
        ).delete()
        self.db.commit()
        return deleted > 0

    def get_conversation_summary(self, conversation_id: int) -> Dict:
        """Get summary information about a conversation."""
        conversation = self.db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            return None
        
        message_count = self.db.query(ConversationMessage).filter(
            ConversationMessage.conversation_id == conversation_id
        ).count()
        
        return {
            "conversation_id": conversation.id,
            "session_id": conversation.session_id,
            "message_count": message_count,
            "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
            "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None,
        }

