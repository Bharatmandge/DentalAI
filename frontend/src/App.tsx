// --- frontend/src/App.tsx ---
import React, { useState } from 'react';
import axios from 'axios';
import { UploadCloud, File, AlertCircle, CheckCircle, Download } from 'lucide-react';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // FastAPI backend URL (Ensure your backend is running on this port!)
  const API_BASE_URL = 'http://127.0.0.1:8000';

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      if (!file.name.endsWith('.nii.gz')) {
        setError("Invalid file type. Please upload a .nii.gz file.");
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
      setError(null);
      setDownloadUrl(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);
    setDownloadUrl(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Send the file to our FastAPI endpoint
      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Construct the full download URL from the backend response
      setDownloadUrl(`${API_BASE_URL}${response.data.mask_download_url}`);
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || "An error occurred during AI processing.";
      setError(errorMsg);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      
      {/* Header Area */}
      <div className="text-center mb-10">
        <h1 className="text-4xl font-extrabold tracking-tight text-white mb-2">
          DentalVision <span className="text-teal-400">AI</span>
        </h1>
        <p className="text-slate-400 text-lg">Clinical 3D CBCT Segmentation Platform</p>
      </div>

      {/* Main Upload Card */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl shadow-2xl p-8 w-full max-w-2xl">
        
        {/* Drag and Drop / File Input */}
        <div className="border-2 border-dashed border-slate-600 rounded-lg p-10 flex flex-col items-center justify-center bg-slate-800/50 hover:bg-slate-700/50 transition-colors">
          <UploadCloud className="w-12 h-12 text-teal-400 mb-4" />
          <p className="text-sm text-slate-300 mb-4 text-center">
            Upload patient CBCT scan to generate a 3D binary mask.
          </p>
          
          <label className="cursor-pointer bg-slate-700 hover:bg-slate-600 text-white font-medium py-2 px-4 rounded-lg transition-colors">
            Select .nii.gz File
            <input type="file" className="hidden" accept=".nii.gz" onChange={handleFileChange} />
          </label>
        </div>

        {/* Selected File Indicator */}
        {selectedFile && (
          <div className="mt-6 p-4 bg-slate-700 rounded-lg flex items-center justify-between border border-slate-600">
            <div className="flex items-center space-x-3">
              <File className="w-5 h-5 text-teal-400" />
              <span className="text-sm font-medium text-slate-200 truncate max-w-xs">{selectedFile.name}</span>
            </div>
            <span className="text-xs text-slate-400">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="mt-6 p-4 bg-red-900/30 border border-red-800 rounded-lg flex items-center space-x-3 text-red-400">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p className="text-sm">{error}</p>
          </div>
        )}

        {/* Success / Download Button */}
        {downloadUrl && (
          <div className="mt-6 p-6 bg-teal-900/20 border border-teal-800 rounded-lg text-center space-y-4">
            <div className="flex justify-center items-center space-x-2 text-teal-400">
              <CheckCircle className="w-6 h-6" />
              <span className="text-lg font-semibold">Segmentation Complete</span>
            </div>
            <a 
              href={downloadUrl}
              className="inline-flex items-center space-x-2 bg-teal-600 hover:bg-teal-500 text-white font-medium py-2 px-6 rounded-lg transition-all shadow-lg hover:shadow-teal-500/25"
            >
              <Download className="w-5 h-5" />
              <span>Download 3D Mask</span>
            </a>
          </div>
        )}

        {/* Action Button */}
        <div className="mt-8">
          <button 
            onClick={handleUpload}
            disabled={!selectedFile || isProcessing}
            className={`w-full py-3 px-4 rounded-lg font-bold text-white transition-all ${
              !selectedFile 
                ? 'bg-slate-700 cursor-not-allowed text-slate-500' 
                : isProcessing 
                  ? 'bg-teal-600/50 cursor-wait' 
                  : 'bg-teal-500 hover:bg-teal-400 shadow-lg shadow-teal-500/20'
            }`}
          >
            {isProcessing ? 'Analyzing Scan via AI Engine...' : 'Run Segmentation'}
          </button>
        </div>

      </div>
    </div>
  );
}

export default App;