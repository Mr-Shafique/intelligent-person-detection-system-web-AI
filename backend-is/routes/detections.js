const express = require('express');
const router = express.Router();
const DetectionLog = require('../models/DetectionLog');
const Person = require('../models/Person'); // Needed to check person status

// GET all detection logs, populated with person details
router.get('/', async (req, res) => {
  try {
    const logs = await DetectionLog.find()
      .sort({ timestamp: -1 }) // Show newest first
      .populate('person', 'name cmsId'); // Populate name and cmsId from the referenced Person
    res.json(logs);
  } catch (error) {
    console.error('Error fetching detection logs:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// POST a new detection log (simulated endpoint for camera system)
// In a real system, this endpoint would receive data from the camera/detection software
router.post('/', async (req, res) => {
  try {
    const { personId, location, capturedImage } = req.body;

    // Validate input
    if (!personId || !location || !capturedImage) {
      return res.status(400).json({ message: 'Missing required fields: personId, location, capturedImage' });
    }

    // Find the person to get their current status
    const person = await Person.findById(personId);
    if (!person) {
      return res.status(404).json({ message: 'Person not found' });
    }

    // Create the new log
    const newLog = new DetectionLog({
      person: personId,
      status: person.status, // Log the status at the time of detection
      location,
      capturedImage,
      timestamp: new Date()
    });

    const savedLog = await newLog.save();
    
    // Populate the person details before sending the response
    const populatedLog = await DetectionLog.findById(savedLog._id)
                                        .populate('person', 'name cmsId');

    res.status(201).json(populatedLog);
  } catch (error) {
    console.error('Error adding detection log:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;