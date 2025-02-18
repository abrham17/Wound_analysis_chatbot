import { Link } from "react-router-dom";
export default function Home() {
  return (
    <div className="relative flex items-center justify-center min-h-screen">
      {/* Background image from Unsplash with an overlay */}
      <div className="absolute inset-0">
        <img 
          src="/medical_image.jpg" 
          alt="Medical Background" 
          className="w-full h-full object-cover" 
        />
        <div className="absolute inset-0 bg-blue-900 opacity-60"></div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 text-center text-white p-6 max-w-2xl">
        <h1 className="text-5xl font-bold mb-4">Wound Analysis Chatbot</h1>
        <p className="text-xl mb-6">
          This innovative tool leverages AI to analyze medical wound images and documents.
          Simply upload an image or document, and receive insights on wound severity, healing progress,
          and related healthcare information.
        </p>
        <Link to="/chatbot">
          <button className="px-8 py-4 bg-green-600 rounded-full shadow-lg hover:bg-green-500 transition">
            Chat with the AI
          </button>
        </Link>
      </div>
    </div>
  );
}
