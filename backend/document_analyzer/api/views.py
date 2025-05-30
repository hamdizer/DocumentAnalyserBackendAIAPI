import asyncio
import json
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_http_methods
from .models import Document, ChatSession, ChatMessage
from .serializers import DocumentSerializer, ChatSessionSerializer, ChatMessageSerializer
from .ai_service import ai_service
import logging
import time

logger = logging.getLogger(__name__)

@api_view(['POST'])
def upload_document(request):
    """
    Upload a document for analysis. Accepts either file upload or text content.
    
    Expected payload:
    {
        "content": "document text content",
        "title": "optional document title"
    }
    """
    try:
        data = request.data
        content = data.get('content', '')
        title = data.get('title', '')
        
        if not content:
            return Response(
                {'error': 'Document content is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create document
        document = Document.objects.create(
            content=content,
            title=title,
            file_size=len(content)
        )
        
        # Create initial chat session
        chat_session = ChatSession.objects.create(document=document)
        
        serializer = DocumentSerializer(document)
        return Response({
            'document': serializer.data,
            'chat_session_id': str(chat_session.id)
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        return Response(
            {'error': f'Failed to upload document: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def process_query(request):
    """
    Process a natural language query against a document.
    
    Expected payload:
    {
        "session_id": "chat_session_uuid",
        "query": "user's natural language query",
        "api_key": "optional API key"
    }
    """
    try:
        data = request.data
        session_id = data.get('session_id')
        query = data.get('query')
        api_key = data.get('api_key')
        
        if not session_id or not query:
            return Response(
                {'error': 'Session ID and query are required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get chat session
        try:
            chat_session = ChatSession.objects.get(id=session_id)
        except ChatSession.DoesNotExist:
            return Response(
                {'error': 'Chat session not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Save user message
        user_message = ChatMessage.objects.create(
            session=chat_session,
            message_type='user',
            content=query
        )
        
        # Process query with AI service
        start_time = time.time()
        
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ai_service.process_query(query, chat_session.document.content, api_key)
            )
        finally:
            loop.close()
        
        processing_time = time.time() - start_time
        
        # Save assistant message
        assistant_message = ChatMessage.objects.create(
            session=chat_session,
            message_type='assistant',
            content=result['response'],
            processing_time=processing_time
        )
        
        return Response({
            'response': result['response'],
            'processing_time': processing_time,
            'message_id': str(assistant_message.id),
            'success': result.get('success', True)
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return Response(
            {'error': f'Failed to process query: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def get_chat_history(request, session_id):
    """
    Get chat history for a specific session.
    """
    try:
        chat_session = ChatSession.objects.get(id=session_id)
        messages = chat_session.messages.all()
        serializer = ChatMessageSerializer(messages, many=True)
        
        return Response({
            'session_id': str(chat_session.id),
            'messages': serializer.data
        })
        
    except ChatSession.DoesNotExist:
        return Response(
            {'error': 'Chat session not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return Response(
            {'error': f'Failed to get chat history: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def health_check(request):
    """
    Health check endpoint to verify system status.
    """
    try:
        # Check AI service status
        ai_status = "ready" if ai_service.is_initialized else "not_initialized"
        
        # Check database connectivity
        db_status = "connected"
        try:
            Document.objects.count()
        except Exception:
            db_status = "error"
        
        return Response({
            'status': 'healthy',
            'ai_service': ai_status,
            'database': db_status,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return Response(
            {'status': 'unhealthy', 'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def generate_summary(request):
    """
    Generate a summary of a document.
    
    Expected payload:
    {
        "document_id": "document_uuid"
    }
    """
    try:
        document_id = request.data.get('document_id')
        
        if not document_id:
            return Response(
                {'error': 'Document ID is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            document = Document.objects.get(id=document_id)
        except Document.DoesNotExist:
            return Response(
                {'error': 'Document not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Generate summary
        summary = ai_service.generate_summary(document.content)
        
        return Response({
            'summary': summary,
            'document_id': str(document.id)
        })
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return Response(
            {'error': f'Failed to generate summary: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )