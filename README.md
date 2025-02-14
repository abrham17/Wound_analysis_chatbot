ChatBot with Image Classification

This is a React-based chatbot that allows users to ask questions about an uploaded wound image. The chatbot first classifies the image using a backend API and then processes the user's question in the context of the classification result.

Features

Users can upload an image (JPEG, PNG, or GIF) under 5MB.

The chatbot classifies the image using a backend API (/api/classify/).

Users can ask questions related to the image, which are sent along with the classification result to a separate API (/api/question/).

Responses are displayed using markdown formatting.

Error handling for failed classification, invalid image uploads, and API failures.

Installation
	
Prerequisites

Node.js and npm installed.

A running backend server that supports the /api/classify/ and /api/question/ endpoints.
	Steps
1/ Clone the repository:
			
	 https://github.com/abrham17/Wound_analysis_chatbot

2/ install frontend dependencies

	cd frontend
 	npm install
	npm start

Backend Installation
Prerequisites

Python installed (recommended: Python 3.8+)
Django and Django REST Framework installed
Pillow library for image processing

Steps
1/ Navigate to the folder

	cd backend
 	python -m venv .venv
	.venv/Scripts/activate
 2/Install dependencies:

	pip install -r requirements.txt
	cd health_assistant

3/Run the database migration

	python manage.py makemigrations
 	python manage.py migrate

4/ Start the Django development server:

	python manage.py runserver

  
  Usage

Click the "Upload Wound Image" button to select an image.

After the image is uploaded, enter a question in the text input field.

Click the "Send" button to submit your question.

The chatbot will classify the image and respond with an answer.

API Endpoints

Backend API

POST /api/classify/

Request: FormData containing an image file.

Response: JSON {'class_name': class_name}

POST /api/question/

Request: JSON JSON.stringify({ question: input,  context: classifiedClass})

Response: JSON  {'question': question, 'answer': generated_answer}

Project Structure
chatbot-project/
	│── frontend/
	│   ├── src/
	│   │   ├── app/dashboard/
	│   │   │   			├── ChatBot.jsx
    |   |   |               ├── Chatbot.css
	│   |	├── App.js
	│   ├── package.json
	│── backend/
    |   ├──health_assistant/
	│   |   ├── api/
	│   │   |      ├── views.py
	│   │   |      ├── serializers.py
	│   │   |      ├── urls.py
 	|   |   ├── health_assistant/
	|	│   ├── manage.py


gate to the backend folder:
