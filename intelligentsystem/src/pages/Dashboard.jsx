import { useState, useEffect } from 'react';
import Card from '../components/Card';
import { api } from '../utils/api';
import Modal from '../components/Modal'; // Import Modal

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalDetections: 0,
    allowedPersons: 0,
    bannedPersons: 0,
    recentDetections: [],
  });
  const [isImageModalOpen, setIsImageModalOpen] = useState(false);
  const [selectedImageUrl, setSelectedImageUrl] = useState('');

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const [personsResponse, logsResponse] = await Promise.all([
        api.getPersons(),
        api.getDetectionLogs(),
      ]);

      const persons = personsResponse.data;
      const logs = logsResponse.data; // logs is an array of DetectionLog documents

      // Process logs to get recent detections with latest event data
      const processedRecentDetections = logs
        .map((log) => {
          if (log.events && log.events.length > 0) {
            // Assuming events are pushed chronologically, the last one is the latest
            const latestEvent = log.events[log.events.length - 1];
            return {
              id: log._id, // Use the main log ID as key for the row
              person_name: log.person_name || 'Unknown',
              timestamp: latestEvent.timestamp,
              status: log.status, // This is the person's status (e.g., allowed, banned)
              location: latestEvent?.camera_source || 'Unknown', // Use the camera source as location
              image_saved: latestEvent?.image_saved, // Add image_saved
            };
          }
          return null; // This log has no events, so it won't be shown in recent detections
        })
        .filter((detection) => detection !== null) // Remove null entries (logs with no events)
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)) // Sort by the latest event's timestamp
        .slice(0, 5); // Get the top 5 recent detections

      setStats({
        totalDetections: logs.length, // Total number of unique persons with logs
        allowedPersons: persons.filter((p) => p.status === 'allowed').length,
        bannedPersons: persons.filter((p) => p.status === 'banned').length,
        recentDetections: processedRecentDetections,
      });
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="p-6">
          <h3 className="text-lg font-medium text-gray-900">Total Detections</h3>
          <p className="mt-2 text-3xl font-semibold text-indigo-600">
            {stats.totalDetections}
          </p>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-medium text-gray-900">Allowed Persons</h3>
          <p className="mt-2 text-3xl font-semibold text-green-600">
            {stats.allowedPersons}
          </p>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-medium text-gray-900">Banned Persons</h3>
          <p className="mt-2 text-3xl font-semibold text-red-600">
            {stats.bannedPersons}
          </p>
        </Card>
      </div>

      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Recent Detections</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Image
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Timestamp
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Location
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {stats.recentDetections.map((detection) => (
                <tr key={detection.id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {detection.image_saved ? (
                      <img
                        src={`http://localhost:5000/${detection.image_saved.replace(/\\\\/g, '/')}`}
                        alt="Detection snapshot"
                        className="h-10 w-10 object-cover border border-gray-200 rounded cursor-pointer"
                        onClick={() => {
                          setSelectedImageUrl(`http://localhost:5000/${detection.image_saved.replace(/\\\\/g, '/')}`);
                          setIsImageModalOpen(true);
                        }}
                        onError={(e) => { e.target.src = 'https://via.placeholder.com/40'; }}
                      />
                    ) : (
                      <div className="h-10 w-10 flex items-center justify-center bg-gray-100 text-gray-400 rounded">No Img</div>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">
                      {detection.person_name}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(detection.timestamp).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        detection.status === 'allowed'
                          ? 'bg-green-100 text-green-800'
                          : detection.status === 'banned'
                          ? 'bg-red-100 text-red-800'
                          : 'bg-gray-100 text-gray-800' // Fallback for other statuses
                      }`}
                    >
                      {detection.status || 'N/A'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {detection.location || 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {isImageModalOpen && (
        <Modal isOpen={isImageModalOpen} onClose={() => setIsImageModalOpen(false)} title="Image Preview">
          <div className="mt-4 flex justify-center items-center">
            <img 
              src={selectedImageUrl} 
              alt="Enlarged detection snapshot" 
              className="max-w-full max-h-[80vh] object-contain"
              onError={(e) => { 
                e.target.alt = 'Error loading image'; 
                // Optionally, display a placeholder or error message within the modal
              }}
            />
          </div>
        </Modal>
      )}
    </div>
  );
};

export default Dashboard;