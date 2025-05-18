const express = require('express');
const router = express.Router();
const DetectionLog = require('../models/DetectionLog');
const Person = require('../models/Person'); // Needed to check person status

// GET all detection logs, grouped by person (like detectionlog.json)
router.get('/', async (req, res) => {
  try {
    const logs = await DetectionLog.find().sort({ 'person_cmsId': 1, 'events.timestamp': -1 });
    res.json(logs);
  } catch (error) {
    console.error('Error fetching detection logs:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// POST a new detection event (append to person or create new person log)
router.post('/', async (req, res) => {
  try {
    const { person_cmsId, person_name, event, status } = req.body; // Add status to destructuring

    if (!person_cmsId || !person_name || !event || !event.timestamp || !event.action || !event.camera_source) {
      return res.status(400).json({ message: 'Missing required fields: person_cmsId, person_name, event (with timestamp, action, camera_source)' });
    }

    // Handle "Unknown" person_cmsId: always create a new log for "Unknown"
    if (person_cmsId === "Unknown") {
      const newLog = new DetectionLog({
        person_cmsId,
        person_name,
        status, // Include status if provided
        events: [event]
      });
      await newLog.save();
      return res.status(201).json(newLog);
    }

    // Try to find an existing log for this person if not "Unknown"
    let log = await DetectionLog.findOne({ person_cmsId });
    if (log) {
      // Append the new event
      log.events.push(event);
      // Update person_name and status if they have changed or are newly provided
      if (person_name && log.person_name !== person_name) {
        log.person_name = person_name;
      }
      if (status && log.status !== status) { // Update status if provided and different
        log.status = status;
      }
      await log.save();
      res.status(200).json(log);
    } else {
      // Create a new log for this person
      const newLog = new DetectionLog({
        person_cmsId,
        person_name,
        status, // Include status if provided
        events: [event]
      });
      await newLog.save();
      res.status(201).json(newLog);
    }
  } catch (error) {
    console.error('Error adding detection log:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;