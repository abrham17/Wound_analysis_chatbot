import React, { useState } from "react";
import ReactMarkdown from 'react-markdown';
import "./chatbot.css";
const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageDataClass, setImageDataClass] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (input.trim() === "") return;

    // Add the user's question as a message
    const userMessage = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);

    try {
      let classifiedClass = null; // To store the classification result

      // Check if an image is uploaded and classify it first
      if (image) {
        const formData = new FormData();
        formData.append("image", image);

        const imageResponse = await fetch("http://localhost:8000/api/classify/", {
          method: "POST",
          body: formData,
        });

        const imageData = await imageResponse.json();
        classifiedClass = imageData.class_name; // Get the class_name from response
        setImageDataClass(classifiedClass); // Update the state
        /*
        setMessages((prev) => [
          ...prev,
          { sender: "bot", text: `Classified as: ${classifiedClass}` },
        ]);*/
      }
      const contextValue = classifiedClass ? classifiedClass : imageDataClass; 
      const questionResponse = await fetch("http://localhost:8000/api/question/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: input,
          context: contextValue, 
        }),
      });

      const questionData = await questionResponse.json();
      console.log(questionData);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `Answer: ${questionData.answer}` },
      ]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Error processing the request." },
      ]);
    }

    setInput("");
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const validTypes = ["image/jpeg", "image/png", "image/gif"];
      if (!validTypes.includes(file.type)) {
        alert("Please upload an image file (jpeg, png, gif).");
        return;
      }

      const maxSize = 5 * 1024 * 1024; // 5MB size limit
      if (file.size > maxSize) {
        alert("File size exceeds 5MB.");
        return;
      }

      setImage(file);

      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="chat-container">
      <div className="image-upload-container" onClick={() => document.getElementById("image-input").click()}>
        <button className="upload-btn">Upload Wound Image</button>
        <input type="file" accept="image/*" id="image-input" style={{ display: "none" }} onChange={handleImageUpload} />
      </div>

      {imagePreview && <img src={imagePreview} alt="Uploaded Preview" className="uploaded-image" />}

      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <ReactMarkdown>{msg.text}</ReactMarkdown>
          </div>
        ))}
      </div>

      <form className="chat-input" onSubmit={handleSubmit}>
        <input type="text" placeholder="Ask something about the image..." value={input} onChange={(e) => setInput(e.target.value)} />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default ChatBot;
