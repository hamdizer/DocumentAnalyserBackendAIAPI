from rest_framework import serializers
from .models import Document, ChatSession, ChatMessage

class DocumentSerializer(serializers.ModelSerializer):
    """
    Serializer for Document model.
    """
    class Meta:
        model = Document
        fields = ['id', 'title', 'content', 'uploaded_at', 'file_size']
        read_only_fields = ['id', 'uploaded_at']

class ChatMessageSerializer(serializers.ModelSerializer):
    """
    Serializer for ChatMessage model.
    """
    class Meta:
        model = ChatMessage
        fields = ['id', 'message_type', 'content', 'timestamp', 'processing_time']
        read_only_fields = ['id', 'timestamp']

class ChatSessionSerializer(serializers.ModelSerializer):
    """
    Serializer for ChatSession model.
    """
    messages = ChatMessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = ChatSession
        fields = ['id', 'document', 'created_at', 'messages']
        read_only_fields = ['id', 'created_at']