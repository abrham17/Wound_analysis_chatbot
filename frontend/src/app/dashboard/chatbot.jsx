// ChatBot.js
import React, { useState, useRef } from 'react';
import { FaFileUpload } from 'react-icons/fa';
import FileUploadSection from './FileUploadSection';
import ChatMessages from './ChatMessages';
import QuestionInput from './QuestionInput';

const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [documentFile, setDocumentFile] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [contextData, setContextData] = useState({});
  const [isFileUploadModalOpen, setIsFileUploadModalOpen] = useState(false);

  const documentInputRef = useRef(null);
  const imageInputRef = useRef(null);

  const handleDocumentChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    if (selectedFile.size > 10 * 1024 * 1024) {
      alert('File size exceeds 10MB limit');
      return;
    }
    setDocumentFile(selectedFile);
  };

  // Handle image selection
  const handleImageChange = (e) => {
    const selectedImage = e.target.files[0];
    if (!selectedImage) return;
    if (selectedImage.size > 10 * 1024 * 1024) {
      alert('File size exceeds 10MB limit');
      return;
    }
    setImageFile(selectedImage);
  };

  // Clear document file
  const clearDocumentFile = () => {
    setDocumentFile(null);
    if (documentInputRef.current) {
      documentInputRef.current.value = null;
    }
  };

  // Clear image file
  const clearImageFile = () => {
    setImageFile(null);
    if (imageInputRef.current) {
      imageInputRef.current.value = null;
    }
  };

  // Drag and Drop handlers for documents
  const handleDocumentDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.size > 10 * 1024 * 1024) {
        alert('File size exceeds 10MB limit');
        return;
      }
      setDocumentFile(file);
      e.dataTransfer.clearData();
    }
  };

  // Drag and Drop handlers for images
  const handleImageDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.size > 10 * 1024 * 1024) {
        alert('File size exceeds 10MB limit');
        return;
      }
      setImageFile(file);
      e.dataTransfer.clearData();
    }
  };

  // Prevent default behavior for drag over
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  // Process document file
  const processDocument = async () => {
    if (!documentFile) return;
    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', documentFile);

    try {
      const response = await fetch('http://localhost:8000/api/process_doc/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error);

      // Update the context with document data (e.g., cache_key)
      setContextData((prev) => ({ ...prev, document: data }));

      setMessages((prev) => [
        ...prev,
        {
          sender: 'system',
          content: `File ${documentFile.name} processed successfully (${data.length} characters)`,
          context: `Document context: ${data.cache_key || 'N/A'}`,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: 'error', content: `Document processing failed: ${error.message}` },
      ]);
    }
    setIsProcessing(false);
  };

  // Process image file
  const processImage = async () => {
    if (!imageFile) return;
    setIsProcessing(true);
    const formData = new FormData();
    // The backend expects the key "image" for image classification
    formData.append('image', imageFile);

    try {
      const response = await fetch('http://localhost:8000/api/classify/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error);

      // Update the context with image classification data
      setContextData((prev) => ({
        ...prev,
        image: { classification: data.classification },
      }));

      setMessages((prev) => [
        ...prev,
        {
          sender: 'system',
          content: `Image ${imageFile.name} classified successfully: ${data.classification}`,
          context: `Classification: ${data.classification}`,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: 'error', content: `Image classification failed: ${error.message}` },
      ]);
    }
    setIsProcessing(false);
  };

  // Handle asking a question
  const handleQuestion = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || !contextData) return;

    // Add the user's question to the messages
    const userMessage = { sender: 'user', content: inputText };
    setMessages((prev) => [...prev, userMessage]);

    // Build the payload including available contexts
    const payload = { question: inputText };
    if (contextData.document && contextData.document.cache_key) {
      payload.cache_key = contextData.document.cache_key;
    }
    if (contextData.image && contextData.image.classification) {
      payload.classification = contextData.image.classification;
    }

    try {
      const response = await fetch('http://localhost:8000/api/question/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error);

      setMessages((prev) => [
        ...prev,
        {
          sender: 'bot',
          content: data.answer,
          context: data.context_used,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: 'error', content: `Error: ${error.message}` },
      ]);
    }
    setInputText('');
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header with ChatBot title and Upload Files button */}
      <div className="p-4 border-b bg-white shadow-sm flex justify-between items-center">
        <h1 className="text-xl font-bold">ChatBot</h1>
        <button
          onClick={() => setIsFileUploadModalOpen(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <FaFileUpload />
          Upload Files
        </button>
      </div>

      {/* Chat Messages Area */}
      <ChatMessages messages={messages} />

      {/* Input Area for Questions */}
      <QuestionInput
        inputText={inputText}
        setInputText={setInputText}
        handleQuestion={handleQuestion}
        contextData={contextData}
      />

      {/* Floating File Upload Modal */}
      {isFileUploadModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center z-50">
          {/* Semi-transparent overlay */}
          <div
            className="absolute inset-0 bg-black opacity-50"
            onClick={() => setIsFileUploadModalOpen(false)}
          ></div>
          <div className="bg-white p-6 rounded-lg shadow-lg z-10 w-11/12 md:w-2/3 lg:w-1/2 relative">
            {/* Close button */}
            <button
              onClick={() => setIsFileUploadModalOpen(false)}
              className="absolute top-2 right-2 text-gray-500 hover:text-gray-700 text-xl"
            >
              &times;
            </button>
            <FileUploadSection
              documentFile={documentFile}
              imageFile={imageFile}
              isProcessing={isProcessing}
              handleDocumentChange={handleDocumentChange}
              handleImageChange={handleImageChange}
              processDocument={processDocument}
              processImage={processImage}
              documentInputRef={documentInputRef}
              imageInputRef={imageInputRef}
              clearDocumentFile={clearDocumentFile}
              clearImageFile={clearImageFile}
              onDocumentDrop={handleDocumentDrop}
              onImageDrop={handleImageDrop}
              onDragOver={handleDragOver}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatBot;
