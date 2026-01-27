from django.urls import path
from . import views

urlpatterns = [
    path('', views.ChatbotView.as_view(), name='chatbot'),
    path('api/chat/', views.ChatbotView.as_view(), name='chat_api'),
    path('api/clear-history/', views.clear_history, name='clear_history'),
]