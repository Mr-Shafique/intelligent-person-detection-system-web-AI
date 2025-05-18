import { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import Card from '../components/Card';
import { api } from '../utils/api';
import Modal from '../components/Modal';

const DetectionLogs = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedLog, setSelectedLog] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isImageModalOpen, setIsImageModalOpen] = useState(false);
  const [selectedImageUrl, setSelectedImageUrl] = useState('');

  useEffect(() => {
    fetchLogs();
  }, []);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const response = await api.getDetectionLogs();
      // Ensure logs is always an array
      setLogs(Array.isArray(response.data) ? response.data : []);
    } catch (error) {
      toast.error('Error fetching detection logs: ' + error.message);
      setLogs([]); // Set to empty array on error
    }
    setLoading(false);
  };

  const handleImageClick = (imageUrl) => {
    setSelectedImageUrl(imageUrl);
    setIsImageModalOpen(true);
  };

  const closeImageModal = () => {
    setIsImageModalOpen(false);
    setSelectedImageUrl('');
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Detection Logs</h1>

      <Card className="p-4">
        <div className="overflow-x-auto">
          {loading ? (
            <p>Loading logs...</p>
          ) : (
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Captured Image
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Person Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    CMS ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status (at detection)
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Location
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Timestamp
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {logs.length === 0 ? (
                  <tr>
                    <td colSpan="7" className="px-6 py-4 text-center text-gray-500">
                      No detection logs found.
                    </td>
                  </tr>
                ) : (
                  logs.map((log) => {
                    const latestEvent = log?.events && log.events.length > 0 
                                      ? log.events[log.events.length - 1] 
                                      : null;
                    return (
                      <tr key={log?._id || Math.random()}>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <img
                            src={latestEvent?.image_saved ? `http://localhost:5000/${latestEvent.image_saved.replace(/\\\\/g, '/')}` : 'https://via.placeholder.com/100'}
                            alt="Detection snapshot"
                            className="h-16 w-16 object-cover border border-gray-200 rounded cursor-pointer"
                            onError={(e) => { e.target.src = 'https://via.placeholder.com/100'; }}
                            onClick={() => latestEvent?.image_saved && handleImageClick(`http://localhost:5000/${latestEvent.image_saved.replace(/\\\\/g, '/')}`)}
                          />
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900">
                            {log?.person_name || 'N/A'}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-500">
                            {log?.person_cmsId || 'N/A'}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span
                            className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                              log?.status === 'allowed'
                                ? 'bg-green-100 text-green-800'
                                : log?.status
                                ? 'bg-red-100 text-red-800' 
                                : 'bg-gray-100 text-gray-800'
                            }`}
                          >
                            {log?.status || 'N/A'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {latestEvent?.camera_source || 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {latestEvent?.timestamp ? new Date(latestEvent.timestamp).toLocaleString() : 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button
                            onClick={() => {
                              setSelectedLog(log);
                              setIsModalOpen(true);
                            }}
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                          >
                            View More
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          )}
        </div>
      </Card>

      {selectedLog && (
        <Modal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} title={`Events for ${selectedLog.person_name}`}>
          <div className="mt-4">
            <h3 className="text-lg font-semibold mb-2">Person: {selectedLog.person_name} (CMS ID: {selectedLog.person_cmsId})</h3>
            {selectedLog.events && selectedLog.events.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Image</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Location</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action (In/Out)</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {selectedLog.events.slice().reverse().map((event, index) => ( // Show latest first
                      <tr key={index}>
                        <td className="px-4 py-2 whitespace-nowrap">
                          {event.image_saved ? (
                            <img
                              src={`http://localhost:5000/${event.image_saved.replace(/\\\\/g, '/')}`}
                              alt="Event snapshot"
                              className="h-12 w-12 object-cover border border-gray-200 rounded cursor-pointer"
                              onError={(e) => { e.target.src = 'https://via.placeholder.com/50'; }}
                              onClick={() => handleImageClick(`http://localhost:5000/${event.image_saved.replace(/\\\\/g, '/')}`)}
                            />
                          ) : (
                            <div className="h-12 w-12 flex items-center justify-center bg-gray-100 text-gray-400 rounded">No Img</div>
                          )}
                        </td>
                        <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-500">{event.camera_source || 'N/A'}</td>
                        <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-500">{event.action || 'N/A'}</td>
                        <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-500">
                          {event.timestamp ? new Date(event.timestamp).toLocaleString() : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p>No events found for this person.</p>
            )}
          </div>
        </Modal>
      )}

      {isImageModalOpen && (
        <Modal isOpen={isImageModalOpen} onClose={closeImageModal} title="Image Preview">
          <div className="mt-4 flex justify-center items-center">
            <img 
              src={selectedImageUrl} 
              alt="Selected detection" 
              className="max-w-full max-h-[80vh] object-contain"
              onError={(e) => { 
                e.target.src = 'https://via.placeholder.com/400?text=Image+Not+Found'; 
                toast.error("Error loading image.");
              }}
            />
          </div>
        </Modal>
      )}
    </div>
  );
};

export default DetectionLogs;