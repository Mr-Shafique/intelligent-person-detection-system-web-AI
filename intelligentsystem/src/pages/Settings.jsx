import { useState, useEffect } from 'react';
import Card from '../components/Card';
import { api } from '../utils/api';

const Settings = () => {
  const [settings, setSettings] = useState({
    soundAlerts: true,
    camera: 'webcam',
    detectionSensitivity: 50,
  });

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const response = await api.getSettings();
      if (response.data) {
        setSettings(response.data);
      }
    } catch (error) {
      console.error('Error fetching settings:', error);
    }
  };

  const handleChange = async (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : value;
    
    const updatedSettings = {
      ...settings,
      [name]: newValue,
    };

    setSettings(updatedSettings);
    await api.updateSettings(updatedSettings);
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Settings</h1>

      <Card className="p-6">
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-medium">Sound Alerts</h3>
              <p className="text-sm text-gray-500">
                Enable or disable sound notifications for detections
              </p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                name="soundAlerts"
                checked={settings.soundAlerts}
                onChange={handleChange}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
            </label>
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
              Camera Source
            </label>
            <select
              name="camera"
              value={settings.camera}
              onChange={handleChange}
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            >
              <option value="webcam">Webcam</option>
              <option value="ip">IP Camera</option>
              <option value="usb">USB Camera</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
              Detection Sensitivity
            </label>
            <input
              type="range"
              name="detectionSensitivity"
              min="0"
              max="100"
              value={settings.detectionSensitivity}
              onChange={handleChange}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-sm text-gray-500">
              <span>Low</span>
              <span>{settings.detectionSensitivity}%</span>
              <span>High</span>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Settings; 