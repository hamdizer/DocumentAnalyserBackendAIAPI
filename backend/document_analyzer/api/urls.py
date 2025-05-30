from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_document, name='upload_document'),
    path('query/', views.process_query, name='process_query'),
    path('chat/<uuid:session_id>/', views.get_chat_history, name='get_chat_history'),
    path('health/', views.health_check, name='health_check'),
    path('summary/', views.generate_summary, name='generate_summary'),
]