import { useState, useEffect, useRef } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Card from '../components/Card';
import { api } from '../utils/api';

const LiveStream = () => {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [detectedPerson, setDetectedPerson] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [persons, setPersons] = useState([]);

  useEffect(() => {
    // Fetch persons from the API when component mounts
    const fetchPersons = async () => {
      try {
        const response = await api.getPersons();
        setPersons(response.data);
      } catch (error) {
        console.error('Error fetching persons:', error);
        toast.error('Error loading person data');
      }
    };
    
    fetchPersons();
    
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  const startStream = async () => {
    try {
      // Request higher resolution
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user' // Prefer front camera
        },
        audio: false,
      };
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setIsStreaming(true);
      setDetectedPerson(null); // Clear previous detection on new stream start
    } catch (err) {
      console.error("Error accessing camera:", err);
      // Try again with default constraints if specific ones fail
      try {
        console.log("Falling back to default camera constraints...");
        const fallbackStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        setStream(fallbackStream);
        if (videoRef.current) {
          videoRef.current.srcObject = fallbackStream;
        }
        setIsStreaming(true);
        setDetectedPerson(null);
      } catch (fallbackErr) {
        toast.error('Error accessing camera: ' + fallbackErr.message);
        console.error("Fallback camera access error:", fallbackErr);
      }
    }
  };

  const stopStream = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setIsStreaming(false);
      setDetectedPerson(null);
    }
  };

  const simulateDetection = () => {
    if (persons.length === 0) {
      toast.warning('No persons in database to simulate detection');
      return;
    }
    
    const randomPerson = persons[Math.floor(Math.random() * persons.length)];
    setDetectedPerson(randomPerson);
    
    if (randomPerson.status === 'banned') {
      toast.error(`⚠️ Banned person detected: ${randomPerson.name}`);
    } else {
      toast.success(`✓ Person detected: ${randomPerson.name}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-center gap-4 mb-4">
        <h1 className="text-2xl font-bold">Live Stream</h1>
        <div className="flex flex-wrap justify-center gap-2">
          {!isStreaming ? (
            <button
              onClick={startStream}
              className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition duration-150 ease-in-out"
            >
              Start Stream
            </button>
          ) : (
            <button
              onClick={stopStream}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition duration-150 ease-in-out"
            >
              Stop Stream
            </button>
          )}
          {isStreaming && (
            <button
              onClick={simulateDetection}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition duration-150 ease-in-out"
            >
              Simulate Detection
            </button>
          )}
        </div>
      </div>

      {/* Combined Card for Video and Detection Info */}
      <Card className="p-4">
        <div className="flex flex-col md:flex-row gap-4">
          {/* Video Feed Area */}
          <div className="flex-grow aspect-video bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
            {!stream && !isStreaming && (
              <div className="w-full h-full flex items-center justify-center text-gray-500">
                Camera feed stopped or not started.
              </div>
            )}
          </div>

          {/* Detected Person Info Area (shown only when detected) */}
          {detectedPerson && (
            <div className="md:w-1/3 lg:w-1/4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <h2 className="text-xl font-semibold mb-4 border-b pb-2">Detected Person</h2>
              <div className="space-y-3">
                {detectedPerson.frontImage ? (
                  <img
                    src={detectedPerson.frontImage}
                    alt={detectedPerson.name}
                    className="w-20 h-20 rounded-full mx-auto border-2 border-gray-300 object-cover"
                  />
                ) : (
                  <div className="w-20 h-20 rounded-full mx-auto flex items-center justify-center bg-gray-200 border-2 border-gray-300 text-gray-500 text-xs">
                    No Image
                  </div>
                )}
                <div className="text-center">
                  <h3 className="text-lg font-medium">{detectedPerson.name}</h3>
                  <p className={`text-sm font-semibold ${
                    detectedPerson.status === 'allowed' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    Status: {detectedPerson.status?.toUpperCase()}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    ID: {detectedPerson.cmsId || 'N/A'}
                  </p>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 text-center border-t pt-2 mt-2">
                  Last seen: {detectedPerson.lastSeen ? new Date(detectedPerson.lastSeen).toLocaleString() : 'N/A'}
                </p>
              </div>
            </div>
          )}
        </div>
      </Card>

      <ToastContainer position="bottom-right" autoClose={3000} theme="colored" />
    </div>
  );
};

export default LiveStream;