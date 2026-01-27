from django.contrib import admin
from .models import ChatHistory

@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ('question', 'category', 'confidence', 'timestamp')
    list_filter = ('category', 'timestamp')
    search_fields = ('question', 'answer')
    readonly_fields = ('timestamp',)