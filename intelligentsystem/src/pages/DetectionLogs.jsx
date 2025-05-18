import { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import Card from '../components/Card';
import { api } from '../utils/api';

const DetectionLogs = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLogs();
  }, []);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const response = await api.getDetectionLogs();
      setLogs(response.data);
    } catch (error) {
      toast.error('Error fetching detection logs: ' + error.message);
    }
    setLoading(false);
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
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {logs.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="px-6 py-4 text-center text-gray-500">
                      No detection logs found.
                    </td>
                  </tr>
                ) : (
                  logs.map((log) => {
                    // Get the latest event for the person
                    // Events are pushed, so the last one is the latest.
                    const latestEvent = log?.events && log.events.length > 0 
                                      ? log.events[log.events.length - 1] 
                                      : null;

                    return (
                      <tr key={log?._id || Math.random()}> {/* Fallback key if log._id is somehow missing */}
                        <td className="px-6 py-4 whitespace-nowrap">
                          <img
                            // src={latestEvent?.image_saved ? `http://localhost:5000/${latestEvent.image_saved.replace(/\\\\\\\\/g, '/')}` : 'https://via.placeholder.com/100'}
                            alt="Detection snapshot"
                            className="h-16 w-16 object-cover border border-gray-200 rounded"
                            onError={(e) => { e.target.src = 'https://via.placeholder.com/100'; }}
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
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          )}
        </div>
      </Card>
    </div>
  );
};

export default DetectionLogs;