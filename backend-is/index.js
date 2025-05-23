require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const connectDB = require('./config/db');
const authRoutes = require('./routes/auth');
const personsRoutes = require('./routes/persons');
const detectionRoutes = require('./routes/detections'); // Import detection routes
const cors = require('cors');
const path = require('path'); // Add path module

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());
// Increase payload size limit (e.g., 50MB)
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Serve static files from the uploads directory
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use(
  '/capturedfaces',
  express.static(path.join(__dirname, '..', 'face-detection-system', 'capturedfaces'))
);

// Connect to MongoDB
connectDB();

const db = mongoose.connection;
db.on('error', console.error.bind(console, 'connection error:'));
db.once('open', () => {
  console.log('Connected to MongoDB');
});

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/persons', personsRoutes);
app.use('/api/detections', detectionRoutes); // Mount detection routes

// Add a route for the root endpoint
app.get('/', (req, res) => {
  res.send('Hello, server is running');
});

// Start the server
app.listen(PORT, () => {
  console.log('Hello, server is running on http://localhost:' + PORT);
});