from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json
from .utils import build_chatbot
from .models import ChatHistory

# Initialize chatbot
chatbot = build_chatbot()

class ChatbotView(View):
    def get(self, request):
        """Render the main chat page"""
        # Get recent chat history
        recent_chats = ChatHistory.objects.all()[:10]
        return render(request, 'chatbot/chat.html', {'recent_chats': recent_chats})
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def post(self, request):
        """Handle chat messages"""
        try:
            data = json.loads(request.body)
            user_question = data.get('question', '').strip()
            
            if not user_question:
                return JsonResponse({'error': 'Question cannot be empty'}, status=400)
            
            # Get response from chatbot
            response = chatbot.chat(user_question)
            
            # Save to chat history
            chat_history = ChatHistory(
                question=user_question,
                answer=response['answer'],
                confidence=response['confidence'],
                category=response['category']
            )
            chat_history.save()
            
            return JsonResponse({
                'answer': response['answer'],
                'confidence': response['confidence'],
                'category': response['category'],
                'matched_question': response.get('matched_question'),
                'related_questions': response.get('related_questions', [])
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

def clear_history(request):
    """Clear chat history"""
    if request.method == 'POST':
        ChatHistory.objects.all().delete()
        return JsonResponse({'success': True})
    return JsonResponse({'error': 'Invalid method'}, status=400)