const express = require('express');
const router = express.Router();
const Person = require('../models/Person');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

// --- Multer Configuration ---
const uploadDir = path.join(__dirname, '..', 'uploads', 'persons');
// Ensure the upload directory exists
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: function(req, file, cb) {
    cb(null, uploadDir);
  },
  filename: function(req, file, cb) {
    // Generate a unique filename: fieldname-timestamp-originalname
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const fileFilter = (req, file, cb) => {
  // Accept only image files
  if (file.mimetype.startsWith('image/')) {
    cb(null, true);
  } else {
    cb(new Error('Not an image! Please upload only images.'), false);
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: { fileSize: 5 * 1024 * 1024 } // 5MB limit
});

// --- End Multer Configuration ---

// Get all persons
router.get('/', async (req, res) => {
  try {
    // Sorting by createdAt here
    const persons = await Person.find().sort({ createdAt: -1 });
    // Convert file paths to URLs
    const personsWithUrls = persons.map(person => {
      const personObj = person.toObject();
      if (personObj.frontImage) {
        personObj.frontImage = `/uploads/persons/${path.basename(personObj.frontImage)}`;
      }
      if (personObj.leftImage) {
        personObj.leftImage = `/uploads/persons/${path.basename(personObj.leftImage)}`;
      }
      if (personObj.rightImage) {
        personObj.rightImage = `/uploads/persons/${path.basename(personObj.rightImage)}`;
      }
      return personObj;
    });
    res.json(personsWithUrls);
  } catch (error) {
    console.error('Error fetching persons:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Add a new person - Apply multer middleware here
// Using upload.fields to handle multiple fields named frontImage, leftImage, rightImage
router.post('/', upload.fields([
  { name: 'frontImage', maxCount: 1 },
  { name: 'leftImage', maxCount: 1 },
  { name: 'rightImage', maxCount: 1 }
]), async (req, res) => {
  console.log('==================== POST /api/persons ====================');
  console.log('req.body:', req.body);
  
  // Create a more terminal-friendly output with clickable links for the files
  if (req.files) {
    console.log('Uploaded Files:');
    Object.keys(req.files).forEach(fieldName => {
      req.files[fieldName].forEach(file => {
        // Convert backslashes to forward slashes for URLs
        const filePath = file.path.replace(/\\/g, '/');
        // Generate a clickable link (works in many terminals)
        // Format: field: filename (size) - file://path/to/file
        console.log(`  ${fieldName}: ${file.originalname} (${(file.size / 1024).toFixed(2)} KB) - file://${filePath}`);
      });
    });
  }
  
  try {
    const { name, cmsId, status } = req.body;
    
    // Validate required fields
    if (!name || !cmsId) {
      return res.status(400).json({ message: 'Name and CMS ID are required' });
    }
    
    // Check if front image was uploaded (required)
    if (!req.files || !req.files.frontImage) {
      return res.status(400).json({ message: 'Front image is required' });
    }
    
    // Get file paths from the uploaded files
    const frontImagePath = req.files.frontImage[0].path;
    const leftImagePath = req.files.leftImage ? req.files.leftImage[0].path : null;
    const rightImagePath = req.files.rightImage ? req.files.rightImage[0].path : null;
    
    // Check if person with the same cmsId already exists
    const existingPerson = await Person.findOne({ cmsId });
    if (existingPerson) {
      // Clean up uploaded files if validation fails
      if (frontImagePath) fs.unlinkSync(frontImagePath);
      if (leftImagePath) fs.unlinkSync(leftImagePath);
      if (rightImagePath) fs.unlinkSync(rightImagePath);
      return res.status(400).json({ message: 'Person with this CMS ID already exists' });
    }
    
    const newPerson = new Person({
      name,
      cmsId,
      status: status || 'allowed',
      frontImage: frontImagePath,
      leftImage: leftImagePath,
      rightImage: rightImagePath
    });
    
    const savedPerson = await newPerson.save();
    
    // Convert paths to URLs for the response
    const personResponse = savedPerson.toObject();
    personResponse.frontImage = `/uploads/persons/${path.basename(personResponse.frontImage)}`;
    if (personResponse.leftImage) {
      personResponse.leftImage = `/uploads/persons/${path.basename(personResponse.leftImage)}`;
    }
    if (personResponse.rightImage) {
      personResponse.rightImage = `/uploads/persons/${path.basename(personResponse.rightImage)}`;
    }
    
    res.status(201).json(personResponse);
  } catch (error) {
    console.error('Error adding person:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Update a person - Apply multer middleware here
router.put('/:id', upload.fields([
  { name: 'frontImage', maxCount: 1 },
  { name: 'leftImage', maxCount: 1 },
  { name: 'rightImage', maxCount: 1 }
]), async (req, res) => {
  console.log('==================== PUT /api/persons/:id ====================');
  console.log('req.body:', req.body);
  
  // Create a more terminal-friendly output with clickable links for the files
  if (req.files && Object.keys(req.files).length > 0) {
    console.log('Uploaded Files for Update:');
    Object.keys(req.files).forEach(fieldName => {
      req.files[fieldName].forEach(file => {
        // Convert backslashes to forward slashes for URLs
        const filePath = file.path.replace(/\\/g, '/');
        // Generate a clickable link (works in many terminals)
        console.log(`  ${fieldName}: ${file.originalname} (${(file.size / 1024).toFixed(2)} KB) - file://${filePath}`);
      });
    });
  } else {
    console.log('No new files uploaded for this update');
  }
  
  try {
    const { name, cmsId, status } = req.body;
    
    // Create update object with text fields
    const updateData = {};
    if (name) updateData.name = name;
    if (cmsId) updateData.cmsId = cmsId;
    if (status) updateData.status = status;
    
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
    
    // Add image paths to updateData if new images were uploaded
    if (req.files) {
      // First find the existing person to get current image paths
      const existingPerson = await Person.findById(req.params.id);
      
      if (!existingPerson) {
        return res.status(404).json({ message: 'Person not found' });
      }
      
      // Handle front image update
      if (req.files.frontImage) {
        // Save the new path
        updateData.frontImage = req.files.frontImage[0].path;
        
        // Delete old image if it exists
        if (existingPerson.frontImage) {
          fs.unlink(existingPerson.frontImage, err => {
            if (err) console.error('Error deleting old front image:', err);
            else console.log(`Deleted old front image: ${existingPerson.frontImage}`);
          });
        }
      }
      
      // Handle left image update
      if (req.files.leftImage) {
        updateData.leftImage = req.files.leftImage[0].path;
        
        if (existingPerson.leftImage) {
          fs.unlink(existingPerson.leftImage, err => {
            if (err) console.error('Error deleting old left image:', err);
            else console.log(`Deleted old left image: ${existingPerson.leftImage}`);
          });
        }
      }
      
      // Handle right image update
      if (req.files.rightImage) {
        updateData.rightImage = req.files.rightImage[0].path;
        
        if (existingPerson.rightImage) {
          fs.unlink(existingPerson.rightImage, err => {
            if (err) console.error('Error deleting old right image:', err);
            else console.log(`Deleted old right image: ${existingPerson.rightImage}`);
          });
        }
      }
    }
    
    const updatedPerson = await Person.findByIdAndUpdate(
      req.params.id,
      updateData,
      { new: true, runValidators: true }
    );
    
    if (!updatedPerson) {
      return res.status(404).json({ message: 'Person not found' });
    }
    
    // Convert paths to URLs for the response
    const personResponse = updatedPerson.toObject();
    if (personResponse.frontImage) {
      personResponse.frontImage = `/uploads/persons/${path.basename(personResponse.frontImage)}`;
    }
    if (personResponse.leftImage) {
      personResponse.leftImage = `/uploads/persons/${path.basename(personResponse.leftImage)}`;
    }
    if (personResponse.rightImage) {
      personResponse.rightImage = `/uploads/persons/${path.basename(personResponse.rightImage)}`;
    }
    
    res.json(personResponse);
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
    
    // Delete associated image files
    if (deletedPerson.frontImage) {
      fs.unlink(deletedPerson.frontImage, err => {
        if (err) console.error('Error deleting front image:', err);
      });
    }
    if (deletedPerson.leftImage) {
      fs.unlink(deletedPerson.leftImage, err => {
        if (err) console.error('Error deleting left image:', err);
      });
    }
    if (deletedPerson.rightImage) {
      fs.unlink(deletedPerson.rightImage, err => {
        if (err) console.error('Error deleting right image:', err);
      });
    }
    
    res.json({ message: 'Person deleted successfully' });
  } catch (error) {
    console.error('Error deleting person:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;