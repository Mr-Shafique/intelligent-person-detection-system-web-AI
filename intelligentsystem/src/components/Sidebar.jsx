import { NavLink } from 'react-router-dom';
import { useState } from 'react';

const Sidebar = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  
  const navItems = [
    { path: '/', label: 'Dashboard', icon: '📊' },
    { path: '/live', label: 'Live Stream', icon: '📹' },
    { path: '/persons', label: 'Person Management', icon: '👥' },
    { path: '/logs', label: 'Detection Logs', icon: '📝' },
    { path: '/settings', label: 'Settings', icon: '⚙️' },
  ];

  return (
    <div className={`bg-gray-800 text-white h-screen fixed left-0 top-0 transition-all duration-300 ${isCollapsed ? 'w-30' : 'w-64'}`}>
      <div className="p-5">
        <div className="flex justify-between items-center mb-8">
          {!isCollapsed && <h2 className="text-2xl font-bold">Menu</h2>}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="text-gray-300 hover:text-white p-2 rounded-lg hover:bg-gray-700"
          >
            {isCollapsed ? '→' : '←'}
          </button>
        </div>
        <nav className="space-y-2">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center px-4 py-2 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-indigo-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700'
                }`
              }
            >
              <span className="mr-3 text-2xl">{item.icon}</span>
              {!isCollapsed && item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </div>
  );
};

export default Sidebar; 