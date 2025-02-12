import { useState, useEffect } from 'react';
import ChatBot from './app/dashboard/chatbot'; 
import './App.css';
function App() {
  return (
    <div>
      <div className="flex flex-1 flex-col gap-4 h-100 w-100 chatbot-fullscreen">
            <ChatBot/>
        </div>
    </div>
  );
}

export default App;
