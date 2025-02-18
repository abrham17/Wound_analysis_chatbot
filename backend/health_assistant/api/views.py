import json
import logging
import os
import dotenv
import PyPDF2
import pandas as pd
from PIL import Image
from docx import Document
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
from datetime import datetime, timedelta

from transformers import AutoImageProcessor, AutoModelForImageClassification
import google.generativeai as genai

dotenv.load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
logger = logging.getLogger(__name__)

# --- Constants ---
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_TYPES = [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'text/csv'
]

# --- In-Memory Document Storage ---
# Each document is stored as a dict with 'content', 'source', and 'timestamp'
documents_store = []

def get_current_documents():
    """Return documents stored within the last hour and remove expired ones."""
    now = datetime.now()
    # Filter valid documents and replace the global list with them
    valid_docs = [doc for doc in documents_store if now - doc['timestamp'] < timedelta(hours=1)]
    documents_store[:] = valid_docs  # Update in place
    return valid_docs

# --- Image Classification Setup ---
image_processor = AutoImageProcessor.from_pretrained("Hemg/Wound-Image-classification")
image_model = AutoModelForImageClassification.from_pretrained("Hemg/Wound-Image-classification")

# --- Custom Gemini LLM Implementation ---
class GeminiLLM:
    def __call__(self, prompt: str) -> str:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text

llm = GeminiLLM()

# --- Utility Function to Extract Text from Files ---
def extract_text_from_file(uploaded_file, content_type):
    text = ""
    try:
        uploaded_file.seek(0)
        if content_type == 'application/pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif content_type in ['text/plain', 'text/csv']:
            text = uploaded_file.read().decode('utf-8')
        elif 'spreadsheet' in content_type or content_type == 'application/vnd.ms-excel':
            try:
                if content_type == 'text/csv':
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                text = df.to_string(index=False)
            except Exception as e:
                logger.error(f"Error reading spreadsheet: {str(e)}")
                text = ""
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
    return text

# --- Django Endpoint: Image Classification ---
@csrf_exempt
def classify_image(request):
    if request.method == 'POST':
        try:
            image_file = request.FILES.get('image')
            if not image_file:
                return JsonResponse({'error': 'Image required'}, status=400)
            img = Image.open(image_file)
            inputs = image_processor(images=img, return_tensors="pt")
            outputs = image_model(**inputs)
            label_id = outputs.logits.argmax(-1).item()
            classification = image_model.config.id2label[label_id]
            confidence = float(outputs.logits.softmax(dim=1)[0][label_id])
            cache_key = f"classification_{image_file.name}"
            cache.set(cache_key, classification, 3600)
            return JsonResponse({
                'classification': classification,
                'confidence': confidence,
                'cache_key': cache_key
            })
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}", exc_info=True)
            return JsonResponse({'error': 'Image processing failed'}, status=500)
    return JsonResponse({'error': 'Invalid method'}, status=405)

# --- Django Endpoint: Document Processing ---
@csrf_exempt
def process_doc(request):
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({'error': 'No file uploaded'}, status=400)
            if uploaded_file.size > MAX_FILE_SIZE:
                return JsonResponse({'error': 'File size exceeds 10MB limit'}, status=400)
            content_type = uploaded_file.content_type
            if content_type not in SUPPORTED_TYPES:
                return JsonResponse({'error': 'Unsupported file type'}, status=400)
            
            text = extract_text_from_file(uploaded_file, content_type)
            if not text.strip():
                return JsonResponse({'error': 'No text extracted from file'}, status=400)
            
            # Store the entire raw text along with its source and timestamp
            documents_store.append({
                'content': text,
                'source': uploaded_file.name,
                'timestamp': datetime.now()
            })
            
            return JsonResponse({'message': 'Document processed and stored successfully'})
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}", exc_info=True)
            return JsonResponse({'error': 'Error processing document'}, status=500)
    return JsonResponse({'error': 'Invalid method'}, status=405)

# --- Django Endpoint: Answering a Question Using Gemini ---
@csrf_exempt
def answer_question(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            question = data.get('question')
            cache_key = data.get('cache_key')
            classification = cache.get(cache_key) if cache_key else None
            
            if not question:
                return JsonResponse({'error': 'Question is required'}, status=400)
            
            valid_docs = get_current_documents()
            text_context = " ".join([doc['content'] for doc in valid_docs]).strip()
            
            combined_context = ""
            prompt = ""
            
            # Both classification and document context available
            if classification and text_context:
                combined_context += f"Wound Image Classification: {classification}\n"
                combined_context += f"Text Data: {text_context}"
                prompt = f"Question: {question}\nContext: {combined_context}\nAnswer:"
            # Only classification available
            elif classification:
                combined_context += f"Wound Image Classification: {classification}\n"
                prompt = f"Context: {combined_context}\nQuestion: {question}\nAnswer:"
            # Only document text available
            elif text_context:
                combined_context += f"Text Data: {text_context}"
                prompt = f"Context: {combined_context}\nQuestion: {question}\nAnswer:"
            else:
                prompt = f"Question: {question}\nAnswer:"
            
            # Generate answer using the LLM
            answer = llm(prompt)
            response_payload = {
                'question': question,
                'answer': answer,
            }
            if combined_context:
                # Limit context string length if too long
                response_payload['context_used'] = (combined_context[:500] + '...') if len(combined_context) > 500 else combined_context
            
            return JsonResponse(response_payload)
        except Exception as e:
            logger.error(f"Answer generation error: {str(e)}", exc_info=True)
            return JsonResponse({'error': 'Error generating answer'}, status=500)
    return JsonResponse({'error': 'Invalid method'}, status=405)
