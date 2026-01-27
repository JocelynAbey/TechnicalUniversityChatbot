"""
Views for chatbot application
"""
import json
import os
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib import messages
import sys

# Add parent directory to path to import predict and train modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import custom database connection
from chatbot_app import dbconnection

# Import chatbot prediction module
try:
    from predict import LBSITWRAGChatbot
    from train import train_chatbot
except ImportError:
    # If imports fail, define fallback
    print("Warning: Could not import predict/train modules. Make sure they are in the project root.")
    LBSITWRAGChatbot = None
    train_chatbot = None

# Initialize chatbot instance
chatbot = None

def get_chatbot():
    """Get or initialize chatbot instance"""
    global chatbot
    if chatbot is None:
        if LBSITWRAGChatbot is None:
            return None
        chatbot = LBSITWRAGChatbot()
        chatbot.load_model(
            settings.CHATBOT_EMBEDDINGS_PATH,
            settings.CHATBOT_DATA_PATH
        )
    return chatbot


def index(request):
    """Public index page with chat interface"""
    return render(request, 'index.html')


@csrf_exempt
def chat_api(request):
    """API endpoint for chat functionality"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_question = data.get('question', '').strip()
            
            if not user_question:
                return JsonResponse({
                    'success': False,
                    'error': 'Please enter a question'
                })
            
            # Get chatbot response
            bot = get_chatbot()
            if bot is None:
                return JsonResponse({
                    'success': False,
                    'error': 'Chatbot model not available. Please contact administrator.'
                })
            
            response = bot.chat(user_question)
            
            return JsonResponse({
                'success': True,
                'answer': response['answer'],
                'confidence': response['confidence'],
                'category': response['category'],
                'matched_question': response.get('matched_question'),
                'related_questions': response.get('related_questions', [])
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


def admin_login(request):
    """Admin login page"""
    if request.session.get('admin_logged_in'):
        return redirect('admin_dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        
        if not username or not password:
            messages.error(request, 'Please enter both username and password')
            return render(request, 'admin/login.html')
        
        # Query database for admin credentials
        qry = f"SELECT * FROM admin WHERE username='{username}' AND password='{password}'"
        result = dbconnection.logindata(qry)
        
        if result:
            request.session['admin_logged_in'] = True
            request.session['admin_username'] = username
            request.session['admin_id'] = result[0]
            messages.success(request, 'Login successful!')
            return redirect('admin_dashboard')
        else:
            messages.error(request, 'Invalid username or password')
    
    return render(request, 'admin/login.html')


def admin_dashboard(request):
    """Admin dashboard"""
    if not request.session.get('admin_logged_in'):
        return redirect('admin_login')
    
    # Get dataset info
    dataset_exists = os.path.exists(settings.CHATBOT_DATASET_PATH)
    model_trained = (os.path.exists(settings.CHATBOT_EMBEDDINGS_PATH) and 
                    os.path.exists(settings.CHATBOT_DATA_PATH))
    
    dataset_count = 0
    if dataset_exists:
        try:
            with open(settings.CHATBOT_DATASET_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                dataset_count = len(data)
        except:
            pass
    
    context = {
        'admin_username': request.session.get('admin_username'),
        'dataset_exists': dataset_exists,
        'model_trained': model_trained,
        'dataset_count': dataset_count
    }
    
    return render(request, 'admin/dashboard.html', context)


def upload_dataset(request):
    """Upload new dataset"""
    if not request.session.get('admin_logged_in'):
        return redirect('admin_login')
    
    if request.method == 'POST':
        if 'dataset_file' not in request.FILES:
            messages.error(request, 'Please select a file to upload')
            return redirect('admin_dashboard')
        
        dataset_file = request.FILES['dataset_file']
        
        if not dataset_file.name.endswith('.json'):
            messages.error(request, 'Please upload a valid JSON file')
            return redirect('admin_dashboard')
        
        try:
            # Read and validate JSON
            file_content = dataset_file.read().decode('utf-8')
            data = json.loads(file_content)
            
            # Validate structure
            if not isinstance(data, list):
                messages.error(request, 'JSON must be an array of Q&A pairs')
                return redirect('admin_dashboard')
            
            for item in data:
                if 'question' not in item or 'answer' not in item:
                    messages.error(request, 'Each item must have "question" and "answer" fields')
                    return redirect('admin_dashboard')
            
            # Save dataset
            with open(settings.CHATBOT_DATASET_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Log upload in database
            admin_id = request.session.get('admin_id')
            qry = f"INSERT INTO dataset_uploads (admin_id, filename, record_count, upload_date) VALUES ({admin_id}, '{dataset_file.name}', {len(data)}, NOW())"
            dbconnection.insertdata(qry)
            
            messages.success(request, f'Dataset uploaded successfully! {len(data)} records loaded.')
            messages.info(request, 'Please train the model to apply changes.')
            
        except json.JSONDecodeError:
            messages.error(request, 'Invalid JSON format')
        except Exception as e:
            messages.error(request, f'Error uploading dataset: {str(e)}')
        
        return redirect('admin_dashboard')
    
    return redirect('admin_dashboard')


def train_model(request):
    """Train the chatbot model"""
    if not request.session.get('admin_logged_in'):
        return redirect('admin_login')
    
    if request.method == 'POST':
        try:
            # Check if dataset exists
            if not os.path.exists(settings.CHATBOT_DATASET_PATH):
                messages.error(request, 'Please upload a dataset first')
                return redirect('admin_dashboard')
            
            # Check if train_chatbot is available
            if train_chatbot is None:
                messages.error(request, 'Training module not available. Check train.py location.')
                return redirect('admin_dashboard')
            
            # Train the model
            success = train_chatbot()
            
            if success:
                # Reload chatbot with new model
                global chatbot
                chatbot = None
                get_chatbot()
                
                # Log training in database
                admin_id = request.session.get('admin_id')
                qry = f"INSERT INTO model_training (admin_id, training_date, status) VALUES ({admin_id}, NOW(), 'success')"
                dbconnection.insertdata(qry)
                
                messages.success(request, 'Model trained successfully!')
            else:
                messages.error(request, 'Model training failed')
                
        except Exception as e:
            messages.error(request, f'Error training model: {str(e)}')
        
        return redirect('admin_dashboard')
    
    return redirect('admin_dashboard')


def admin_logout(request):
    """Logout admin"""
    request.session.flush()
    messages.success(request, 'Logged out successfully')
    return redirect('admin_login')