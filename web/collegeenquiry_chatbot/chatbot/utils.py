from college_enquiry_chatbot.config import data_path, model_path
from college_enquiry_chatbot.core.rag import CollegeEnquiryRAGChatbot


def build_chatbot() -> CollegeEnquiryRAGChatbot:
    chatbot = CollegeEnquiryRAGChatbot(
        data_path=data_path(),
        model_path=model_path(),
    )
    chatbot.load_model()
    return chatbot