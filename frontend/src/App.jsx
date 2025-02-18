import { BrowserRouter as Router , Route , Routes} from 'react-router-dom';
import React from 'react';
import { useState, useEffect } from 'react';
import ChatBot from './app/dashboard/chatbot'; 
import Home from './components/home';
import './App.css';
function App() {
  return (
      <Router>
        <Routes>
          <Route path="/" element={<Home/>} />
          <Route path="chatbot/" element={<ChatBot/>} />
        </Routes>
      </Router>
  );
}

export default App;
