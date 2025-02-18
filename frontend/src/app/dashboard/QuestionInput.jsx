// QuestionInput.js
import React from 'react';
import { FaPaperPlane } from 'react-icons/fa';

const QuestionInput = ({ inputText, setInputText, handleQuestion, contextData }) => {
  return (
    <form onSubmit={handleQuestion} className="p-4 border-t bg-white">
      <div className="flex gap-4">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Ask a question based on your document and/or image..."
          className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          type="submit"
          className="p-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
        >
          <FaPaperPlane />
        </button>
      </div>
    </form>
  );
};

export default QuestionInput;
