const express = require('express');
const router = express.Router();
const Person = require('../models/Person');

// Get all persons
router.get('/', async (req, res) => {
  try {
    // Sorting by createdAt here
    const persons = await Person.find().sort({ createdAt: -1 });
    res.json(persons);
  } catch (error) {
    console.error('Error fetching persons:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Add a new person
router.post('/', async (req, res) => {
  try {
    const { name, cmsId, image, status } = req.body;
    
    // Check if person with the same cmsId already exists
    const existingPerson = await Person.findOne({ cmsId });
    if (existingPerson) {
      return res.status(400).json({ message: 'Person with this CMS ID already exists' });
    }
    
    const newPerson = new Person({
      name,
      cmsId,
      image,
      status,
      lastSeen: new Date()
    });
    
    const savedPerson = await newPerson.save();
    res.status(201).json(savedPerson);
  } catch (error) {
    console.error('Error adding person:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Update a person
router.put('/:id', async (req, res) => {
  try {
    const { name, cmsId, image, status } = req.body;
    
    // Check if another person with the same cmsId already exists
    if (cmsId) {
      const existingPerson = await Person.findOne({ 
        cmsId, 
        _id: { $ne: req.params.id } 
      });
      
      if (existingPerson) {
        return res.status(400).json({ message: 'Another person with this CMS ID already exists' });
      }
    }
    
    const updatedPerson = await Person.findByIdAndUpdate(
      req.params.id,
      { name, cmsId, image, status },
      { new: true }
    );
    
    if (!updatedPerson) {
      return res.status(404).json({ message: 'Person not found' });
    }
    
    res.json(updatedPerson);
  } catch (error) {
    console.error('Error updating person:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Delete a person
router.delete('/:id', async (req, res) => {
  try {
    const deletedPerson = await Person.findByIdAndDelete(req.params.id);
    
    if (!deletedPerson) {
      return res.status(404).json({ message: 'Person not found' });
    }
    
    res.json({ message: 'Person deleted successfully' });
  } catch (error) {
    console.error('Error deleting person:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;