// FileUploadSection.js
import React from 'react';
import { FaFileUpload, FaImage, FaTrash } from 'react-icons/fa';

const FileUploadSection = ({
  documentFile,
  imageFile,
  isProcessing,
  handleDocumentChange,
  handleImageChange,
  processDocument,
  processImage,
  documentInputRef,
  imageInputRef,
  clearDocumentFile,
  clearImageFile,
  onDocumentDrop,
  onImageDrop,
  onDragOver,
}) => {
  return (
    <div className="flex flex-col gap-6">
      <h2 className="text-lg font-bold mb-2">Upload Files</h2>
      <div className="flex flex-col gap-4 md:flex-row md:justify-around">
        {/* Document Upload Drop Zone */}
        <div
          onDrop={onDocumentDrop}
          onDragOver={onDragOver}
          className="relative border-dashed border-2 border-blue-300 p-4 rounded-lg flex-1 flex flex-col items-center"
          style={{ minHeight: '200px' }}
        >
          <input
            type="file"
            ref={documentInputRef}
            onChange={handleDocumentChange}
            className="hidden"
            accept=".pdf,.docx,.txt,.csv,.xlsx"
          />
          <button
            onClick={() => documentInputRef.current.click()}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <FaFileUpload />
            {documentFile ? documentFile.name : 'Upload Document'}
          </button>
          {documentFile && (
            <div className="mt-2 text-sm text-gray-600 text-center">
              <p>Type: {documentFile.type || 'N/A'}</p>
              <p>Size: {(documentFile.size / 1024).toFixed(2)} KB</p>
              <button
                onClick={clearDocumentFile}
                className="mt-1 text-red-500 hover:text-red-700 flex items-center gap-1"
              >
                <FaTrash /> Remove Document
              </button>
            </div>
          )}
          {documentFile && (
            <button
              onClick={processDocument}
              disabled={isProcessing}
              className="mt-2 w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400"
            >
              {isProcessing ? 'Processing...' : 'Process Document'}
            </button>
          )}
        </div>

        {/* Image Upload Drop Zone */}
        <div
          onDrop={onImageDrop}
          onDragOver={onDragOver}
          className="relative border-dashed border-2 border-blue-300 p-4 rounded-lg flex-1 flex flex-col items-center"
          style={{ minHeight: '200px' }}
        >
          <input
            type="file"
            ref={imageInputRef}
            onChange={handleImageChange}
            className="hidden"
            accept="image/*"
          />
          <button
            onClick={() => imageInputRef.current.click()}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <FaImage />
            {imageFile ? imageFile.name : 'Upload Image'}
          </button>
          {imageFile && (
            <div className="mt-2 text-center">
              <img
                src={URL.createObjectURL(imageFile)}
                alt="Preview"
                className="w-32 h-32 object-cover rounded-lg border mx-auto"
              />
              <div className="mt-1 text-sm text-gray-600">
                <p>Size: {(imageFile.size / 1024).toFixed(2)} KB</p>
                <button
                  onClick={clearImageFile}
                  className="mt-1 text-red-500 hover:text-red-700 flex items-center gap-1 justify-center"
                >
                  <FaTrash /> Remove Image
                </button>
              </div>
            </div>
          )}
          {imageFile && (
            <button
              onClick={processImage}
              disabled={isProcessing}
              className="mt-2 w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400"
            >
              {isProcessing ? 'Processing...' : 'Classify Image'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileUploadSection;
