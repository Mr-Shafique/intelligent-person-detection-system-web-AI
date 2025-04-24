export const mockPersons = [
  {
    id: 1,
    name: 'John Doe',
    status: 'allowed',
    image: 'https://via.placeholder.com/150',
    lastSeen: '2024-03-20T10:30:00',
  },
  {
    id: 2,
    name: 'Jane Smith',
    status: 'banned',
    image: 'https://via.placeholder.com/150',
    lastSeen: '2024-03-20T11:15:00',
  },
  {
    id: 3,
    name: 'Mike Johnson',
    status: 'allowed',
    image: 'https://via.placeholder.com/150',
    lastSeen: '2024-03-20T12:00:00',
  },
];

export const mockDetectionLogs = [
  {
    id: 1,
    personId: 1,
    name: 'John Doe',
    timestamp: '2024-03-20T10:30:00',
    status: 'allowed',
    location: 'Main Entrance',
  },
  {
    id: 2,
    personId: 2,
    name: 'Jane Smith',
    timestamp: '2024-03-20T11:15:00',
    status: 'banned',
    location: 'Security Room',
  },
  {
    id: 3,
    personId: 3,
    name: 'Mike Johnson',
    timestamp: '2024-03-20T12:00:00',
    status: 'allowed',
    location: 'Main Hall',
  },
  {
    id: 4,
    personId: 1,
    name: 'John Doe',
    timestamp: '2024-03-20T13:30:00',
    status: 'allowed',
    location: 'Main Entrance',
  },
  {
    id: 5,
    personId: 2,
    name: 'Jane Smith',
    timestamp: '2024-03-20T14:15:00',
    status: 'banned',
    location: 'Security Room',
  },
]; 