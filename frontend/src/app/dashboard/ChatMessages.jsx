// ChatMessages.js
import React from 'react';
import ReactMarkdown from 'react-markdown';

const ChatMessages = ({ messages }) => {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((msg, index) => (
        <div
          key={index}
          className={`p-4 rounded-lg max-w-3xl ${
            msg.sender === 'user'
              ? 'ml-auto bg-blue-100 border-blue-200'
              : msg.sender === 'bot'
              ? 'bg-white border-gray-200 shadow'
              : 'bg-red-100 border-red-200'
          } border`}
        >
          <ReactMarkdown className="prose">{msg.content}</ReactMarkdown>
          {msg.context && (
            <div className="mt-2 text-sm text-gray-500">
              Context used: {msg.context}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default ChatMessages;
