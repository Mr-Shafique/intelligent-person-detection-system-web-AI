import { useState, useEffect, useRef } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Card from '../components/Card';
import Loader from '../components/Loader';
import { api } from '../utils/api';

const LiveStream = () => {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [detectedPerson, setDetectedPerson] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [persons, setPersons] = useState([]);
  const [recentDetections, setRecentDetections] = useState([]);
  const [loadingDetections, setLoadingDetections] = useState(false);
  const [detectedPersonEvent, setDetectedPersonEvent] = useState(null);

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

  const startStream = () => {
    setIsStreaming(true);
    setDetectedPerson(null);
  };

  const stopStream = () => {
    setIsStreaming(false);
    setDetectedPerson(null);
  };

  const fetchRecentDetections = async () => {
    setLoadingDetections(true);
    try {
      const logsResponse = await api.getDetectionLogs();
      const logs = logsResponse.data;
      // Process logs to get recent detections with latest event data (same as Dashboard)
      const processedRecentDetections = logs
        .map((log) => {
          if (log.events && log.events.length > 0) {
            const latestEvent = log.events[log.events.length - 1];
            return {
              id: log._id,
              person_name: log.person_name || 'Unknown',
              timestamp: latestEvent.timestamp,
              status: log.status,
              location: latestEvent?.camera_source || 'Unknown',
              image_saved: latestEvent?.image_saved,
            };
          }
          return null;
        })
        .filter((detection) => detection !== null)
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
        .slice(0, 5);
      setRecentDetections(processedRecentDetections);
    } catch (error) {
      setRecentDetections([]);
    }
    setLoadingDetections(false);
  };

  const fetchLatestEventForPerson = async (person) => {
    try {
      const logsResponse = await api.getDetectionLogs();
      const logs = logsResponse.data;
      // Find the log for this person (by name or unique id)
      const log = logs.find(
        (l) => l.person_name === person.name || l.person_name === person.cmsId || l.person_id === person._id
      );
      if (log && log.events && log.events.length > 0) {
        const latestEvent = log.events[log.events.length - 1];
        setDetectedPersonEvent({
          timestamp: latestEvent.timestamp,
          status: log.status,
          location: latestEvent?.camera_source || 'Unknown',
          image_saved: latestEvent?.image_saved,
        });
      } else {
        setDetectedPersonEvent(null);
      }
    } catch (error) {
      setDetectedPersonEvent(null);
    }
  };

  const simulateDetection = () => {
    if (persons.length === 0) {
      toast.warning('No persons in database to simulate detection');
      return;
    }

    const randomPerson = persons[Math.floor(Math.random() * persons.length)];
    setDetectedPerson(randomPerson);
    fetchLatestEventForPerson(randomPerson);

    if (randomPerson.status === 'banned') {
      toast.error(`⚠️ Banned person detected: ${randomPerson.name}`);
    } else {
      toast.success(`✓ Person detected: ${randomPerson.name}`);
    }

    // Fetch recent detections after simulating
    fetchRecentDetections();
  };

  // Optionally, fetch on mount as well
  useEffect(() => {
    fetchRecentDetections();
  }, []);

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
            {isStreaming ? (
              <img
                src="http://localhost:5002/video_feed"
                alt="Live Stream"
                className="w-full h-full object-cover"
                style={{ minHeight: 240, background: "#222" }}
              />
            ) : (
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
                {detectedPersonEvent.image_saved ? (
                    <img
                        src={`http://localhost:5000/${detectedPersonEvent.image_saved.replace(/\\/g, '/')}`}
                        alt="Event snapshot"
                        className="w-16 h-16 object-cover rounded border mx-auto mb-2"
                        onError={(e) => { e.target.src = 'https://via.placeholder.com/40'; }}
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
                {/* Latest detection event details */}
                {detectedPersonEvent && (
                  <div className="mt-4 p-2 bg-gray-100 dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-700">
                    <h4 className="text-sm font-semibold mb-2">Latest Detection Event</h4>
                    {detectedPersonEvent.image_saved ? (
                      <img
                        src={`http://localhost:5000/${detectedPersonEvent.image_saved.replace(/\\/g, '/')}`}
                        alt="Event snapshot"
                        className="w-16 h-16 object-cover rounded border mx-auto mb-2"
                        onError={(e) => { e.target.src = 'https://via.placeholder.com/40'; }}
                      />
                    ) : (
                      <div className="w-16 h-16 flex items-center justify-center bg-gray-200 text-gray-400 rounded mx-auto mb-2">No Img</div>
                    )}
                    <div className="text-xs text-gray-700 dark:text-gray-300">
                      <div><span className="font-semibold">Time:</span> {new Date(detectedPersonEvent.timestamp).toLocaleString()}</div>
                      <div><span className="font-semibold">Status:</span> {detectedPersonEvent.status || 'N/A'}</div>
                      <div><span className="font-semibold">Location:</span> {detectedPersonEvent.location || 'N/A'}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Recent Detections Card */}
      <Card className="p-6 mt-6">
        <h2 className="text-xl font-semibold mb-4">Recent Detections</h2>
        <div className="overflow-x-auto">
          {loadingDetections ? (
            <Loader />
          ) : (
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Image</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Location</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {recentDetections.length === 0 ? (
                  <tr>
                    <td colSpan="5" className="px-6 py-4 text-center text-gray-500">No detection logs found.</td>
                  </tr>
                ) : (
                  recentDetections.map((detection) => (
                    <tr key={detection.id}>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {detection.image_saved ? (
                          <img
                            src={`http://localhost:5000/${detection.image_saved.replace(/\\/g, '/')}`}
                            alt="Detection snapshot"
                            className="h-10 w-10 object-cover border border-gray-200 rounded"
                            onError={(e) => { e.target.src = 'https://via.placeholder.com/40'; }}
                          />
                        ) : (
                          <div className="h-10 w-10 flex items-center justify-center bg-gray-100 text-gray-400 rounded">No Img</div>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">{detection.person_name}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(detection.timestamp).toLocaleString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          detection.status === 'allowed'
                            ? 'bg-green-100 text-green-800'
                            : detection.status === 'banned'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}>
                          {detection.status || 'N/A'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {detection.location || 'N/A'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          )}
        </div>
      </Card>

      <ToastContainer position="bottom-right" autoClose={3000} theme="colored" />
    </div>
  );
};

export default LiveStream;