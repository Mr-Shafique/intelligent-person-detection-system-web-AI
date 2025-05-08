import { useState, useEffect } from 'react';
import Card from '../components/Card';
import { api } from '../utils/api';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalDetections: 0,
    allowedPersons: 0,
    bannedPersons: 0,
    recentDetections: [],
  });

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
      const logs = logsResponse.data;

      setStats({
        totalDetections: logs.length,
        allowedPersons: persons.filter((p) => p.status === 'allowed').length,
        bannedPersons: persons.filter((p) => p.status === 'banned').length,
        recentDetections: logs.slice(0, 5),
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
        <h2 className="text-xl font-semibold mb-4">Recent  Detections</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
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
              {stats.recentDetections.map((log) => (
                <tr key={log.id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">
                      {log.person.name}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(log.timestamp).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        log.status === 'allowed'
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}
                    >
                      {log.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {log.location}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};

export default Dashboard; 