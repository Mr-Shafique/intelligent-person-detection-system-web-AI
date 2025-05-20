import { useState, useEffect } from 'react';
import Loader from '../components/Loader';
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
  const [searchTerm, setSearchTerm] = useState('');
  const [dateFilter, setDateFilter] = useState('');
  const [selectedAnalysisLog, setSelectedAnalysisLog] = useState(null);
  const [isAnalysisModalOpen, setIsAnalysisModalOpen] = useState(false);
  const [analysisDate, setAnalysisDate] = useState('');


  useEffect(() => {
    fetchLogs();
    const interval = setInterval(() => {
      fetchLogs();
    }, 5000);
    return () => clearInterval(interval);
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

  // Filter logs by search term and date
  const filteredLogs = logs.filter((log) => {
    const nameMatch = log?.person_name?.toLowerCase().includes(searchTerm.toLowerCase());
    const cmsIdMatch = log?.person_cmsId?.toLowerCase().includes(searchTerm.toLowerCase());
    // Check if any event matches the date filter
    let dateMatch = true;
    if (dateFilter) {
      dateMatch = log.events && log.events.some(event => {
        if (!event.timestamp) return false;
        const eventDate = new Date(event.timestamp).toISOString().slice(0, 10);
        return eventDate === dateFilter;
      });
    }
    return (nameMatch || cmsIdMatch) && dateMatch;
  });

  function getAnalysisForDate(log, date) {
    if (!log || !log.events) return {};

    // Filter events for the selected date
    const events = log.events.filter(ev => {
      if (!ev.timestamp) return false;
      const eventDate = new Date(ev.timestamp).toISOString().slice(0, 10);
      return eventDate === date;
    }).sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

    // Group by block
    const blocks = {};
    events.forEach(ev => {
      const block = ev.camera_source || 'Unknown';
      if (!blocks[block]) {
        blocks[block] = { in: 0, out: 0, times: [], totalTimeMs: 0 };
      }
      if (ev.action === 'IN') {
        blocks[block].in += 1;
        blocks[block].times.push({ in: new Date(ev.timestamp), out: null });
      } else if (ev.action === 'OUT') {
        blocks[block].out += 1;
        // Pair with last unmatched IN
        const last = blocks[block].times.find(t => t.out === null);
        if (last) last.out = new Date(ev.timestamp);
      }
    });

    // Calculate total time spent per block
    Object.values(blocks).forEach(block => {
      block.totalTimeMs = block.times.reduce((sum, t) => {
        if (t.in && t.out) return sum + (t.out - t.in);
        return sum;
      }, 0);
    });

    return blocks;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Detection Logs</h1>

      {/* Search and filter bar */}
      <div className="flex flex-col md:flex-row md:items-center gap-4 mb-4">
        <div className="flex-1">
          <input
            type="text"
            placeholder="Search by Name or CMS ID..."
            className="border border-gray-300 rounded px-4 py-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-400"
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="w-full md:w-1/4">
          <input
            type="date"
            className="border border-gray-300 rounded px-4 py-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-400 cursor-pointer"
            value={dateFilter}
            onChange={e => setDateFilter(e.target.value)}
            onFocus={e => e.target.showPicker && e.target.showPicker()}
            onClick={e => e.target.showPicker && e.target.showPicker()}
            style={{ minWidth: 0 }} // ensures it doesn't overflow its container
          />
        </div>
      </div>

      <Card className="p-4">
        <div className="overflow-x-auto">
          {loading ? (
            <Loader />
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
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Detailed Analysis
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredLogs.length === 0 ? (
                  <tr>
                    <td colSpan="8" className="px-6 py-4 text-center text-gray-500">
                      No detection logs found.
                    </td>
                  </tr>
                ) : (
                  filteredLogs.map((log) => {
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
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button
                            onClick={() => {
                              setSelectedAnalysisLog(log);
                              setIsAnalysisModalOpen(true);
                            }}
                            className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded"
                          >
                            Detailed Analysis
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
          <div className="mt-4 w-full max-w-6xl mx-auto">
            <h3 className="text-2xl font-semibold mb-4 text-center">
              Person: {selectedLog.person_name} (CMS ID: {selectedLog.person_cmsId})
            </h3>
            {selectedLog.events && selectedLog.events.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 text-base">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider">Image</th>
                      <th className="px-6 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider">Location</th>
                      <th className="px-6 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider">IN/OUT</th>
                      <th className="px-6 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {selectedLog.events.slice().reverse().map((event, index) => (
                      <tr key={index}>
                        <td className="px-6 py-3 whitespace-nowrap">
                          {event.image_saved ? (
                            <img
                              src={`http://localhost:5000/${event.image_saved.replace(/\\\\/g, '/')}`}
                              alt="Event snapshot"
                              className="h-24 w-24 object-cover border border-gray-200 rounded cursor-pointer transition-transform duration-200 hover:scale-110"
                              onError={(e) => { e.target.src = 'https://via.placeholder.com/80'; }}
                              onClick={() => handleImageClick(`http://localhost:5000/${event.image_saved.replace(/\\\\/g, '/')}`)}
                            />
                          ) : (
                            <div className="h-24 w-24 flex items-center justify-center bg-gray-100 text-gray-400 rounded">No Img</div>
                          )}
                        </td>
                        <td className="px-6 py-3 whitespace-nowrap text-base text-gray-700">{event.camera_source || 'N/A'}</td>
                        <td className="px-6 py-3 whitespace-nowrap text-base text-gray-700">{event.action === 'IN' ? 'IN' : event.action === 'OUT' ? 'OUT' : 'N/A'}</td>
                        <td className="px-6 py-3 whitespace-nowrap text-base text-gray-700">
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
          <div className=" flex justify-center items-center">
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

      {selectedAnalysisLog && (
        <Modal isOpen={isAnalysisModalOpen} onClose={() => setIsAnalysisModalOpen(false)} title={`Detailed Analysis for ${selectedAnalysisLog.person_name}`}>
          <div className="mb-4">
            <input
              type="date"
              value={analysisDate}
              onChange={e => setAnalysisDate(e.target.value)}
              className="border border-gray-300 rounded px-4 py-2"
            />
          </div>
          {analysisDate ? (
            (() => {
              const blocks = getAnalysisForDate(selectedAnalysisLog, analysisDate);
              const blockNames = Object.keys(blocks);
              if (blockNames.length === 0) return <p>No data for this date.</p>;
              return (
                <table className="min-w-full divide-y divide-gray-200">
                  <thead>
                    <tr>
                      <th>Block</th>
                      <th>Entries (IN)</th>
                      <th>Exits (OUT)</th>
                      <th>Total Time Spent</th>
                    </tr>
                  </thead>
                  <tbody>
                    {blockNames.map(block => (
                      <tr key={block}>
                        <td>{block}</td>
                        <td>{blocks[block].in}</td>
                        <td>{blocks[block].out}</td>
                        <td>
                          {blocks[block].totalTimeMs > 0
                            ? new Date(blocks[block].totalTimeMs).toISOString().substr(11, 8)
                            : '0:00:00'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              );
            })()
          ) : (
            <p>Please select a date.</p>
          )}
        </Modal>
      )}
    </div>
  );
};

export default DetectionLogs;