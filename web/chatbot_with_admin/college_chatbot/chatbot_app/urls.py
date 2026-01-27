"""
URL patterns for chatbot_app
"""
from django.urls import path
from . import views

urlpatterns = [
    # Public routes
    path('', views.index, name='index'),
    path('chat/', views.chat_api, name='chat_api'),
    
    # Admin routes
    path('admin/login/', views.admin_login, name='admin_login'),
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin/upload-dataset/', views.upload_dataset, name='upload_dataset'),
    path('admin/train-model/', views.train_model, name='train_model'),
    path('admin/logout/', views.admin_logout, name='admin_logout'),
]