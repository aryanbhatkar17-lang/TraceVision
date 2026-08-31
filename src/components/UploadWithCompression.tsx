'use client';

import React, { useState } from 'react';
import { compressVideo } from '@/lib/compress';

export default function UploadWithCompression() {
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [isCompressing, setIsCompressing] = useState<boolean>(false);
  const [compressedBlob, setCompressedBlob] = useState<Blob | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleCompress = async () => {
    if (!file) return;

    setIsCompressing(true);
    setProgress(0);
    setCompressedBlob(null);

    try {
      const result = await compressVideo({
        file,
        profileKey: 'balanced',
        onProgress: (p) => setProgress(p),
      });
      setCompressedBlob(result.blob);
    } catch (error) {
      console.error('Compression failed:', error);
    } finally {
      setIsCompressing(false);
    }
  };

  const handleUpload = async () => {
    if (!compressedBlob) return;
    
    // Simulating the upload to FastAPI backend
    const formData = new FormData();
    const fileName = file?.name.replace(/\.[^/.]+$/, "") + '_compressed.mp4';
    formData.append('file', compressedBlob, fileName);
    
    alert(`Ready to upload! Compressed size: ${(compressedBlob.size / 1024 / 1024).toFixed(2)} MB`);
    // await fetch('/api/upload', { method: 'POST', body: formData });
  };

  return (
    <div className="p-6 max-w-md mx-auto bg-white rounded-xl shadow-md space-y-4 border border-gray-200 mt-10">
      <h2 className="text-xl font-bold text-gray-900">TraceVision Upload</h2>
      
      <input 
        type="file" 
        accept="video/*" 
        onChange={handleFileChange} 
        disabled={isCompressing}
        className="block w-full text-sm text-gray-500
          file:mr-4 file:py-2 file:px-4
          file:rounded-md file:border-0
          file:text-sm file:font-semibold
          file:bg-blue-50 file:text-blue-700
          hover:file:bg-blue-100"
      />

      {file && (
        <button
          onClick={handleCompress}
          disabled={isCompressing}
          className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isCompressing ? `Compressing (${progress}%)` : 'Compress Video'}
        </button>
      )}

      {isCompressing && (
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" 
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      )}

      {compressedBlob && !isCompressing && (
        <div className="space-y-3">
          <div className="text-sm p-3 bg-green-50 text-green-700 rounded-md">
            <p className="font-semibold">Compression complete!</p>
            <p>Original: {(file!.size / 1024 / 1024).toFixed(2)} MB</p>
            <p>Compressed: {(compressedBlob.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
          <button
            onClick={handleUpload}
            className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
          >
            Upload to Backend
          </button>
        </div>
      )}
    </div>
  );
}
