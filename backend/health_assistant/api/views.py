from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms, models
import google.generativeai as genai
import json
import os
from huggingface_hub import hf_hub_download

# Download the model
from torchvision import models

class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):  # Adjust num_classes as needed
        super(CustomResNet, self).__init__()
        self.base_model = models.resnet50(pretrained=False)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

model_path = hf_hub_download(repo_id="Abrhame/resnet-image-classifier-v2", filename="pytorch_model.bin")

# Load the model
model = CustomResNet(num_classes=7)
model.load_state_dict(torch.load(model_path) , strict=False)
model.eval()
# Load Google Gemini API key
GOOGLE_API_KEY = "AIzaSyBnIUvvrrMPMDUshQNTBfEEZLzhPtiggTA"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)
class_names = ["Abrasion", "Bruises", "Burn", "Cut", "Fracture", "Ingrown_nails", "Laceration"]

# Load ResNet model for image classification

"""
model = models.resnet50(pretrained=False)
num_classes = 7
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load saved weights
model.load_state_dict(torch.load(
    "C:\\Users\\yonas\\OneDrive\\Desktop\\projects\\MLB\\MLB_Prediction\\backend\\health_assistant\\api\\best_model.pth",
    map_location='cpu'
))
model.eval()

"""
# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@csrf_exempt
def classify_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')

        if not image_file:
            return JsonResponse({'error': 'Image is required.'}, status=400)

        try:
            image = Image.open(image_file).convert("RGB")
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

            class_name = class_names[predicted.item()]
            confidence_score = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()

            response_data = {
                'class_name': class_name,
                'confidence': confidence_score
            }

            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)


@csrf_exempt
def answer_question(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            question = data.get('question')
            context = data.get('context', '')

            if not question:
                return JsonResponse({'error': 'Question is required.'}, status=400)

            input_text = f"Question: {question}\nContext: {context}"

            # Gemini API Call
            model = genai.GenerativeModel('gemini-pro')  # Using Gemini Pro model
            response = model.generate_content(input_text)

            generated_answer = response.text.strip()  # Extract the generated response

            return JsonResponse({'question': question, 'answer': generated_answer})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)
