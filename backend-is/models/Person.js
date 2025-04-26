const mongoose = require('mongoose');

const PersonSchema = new mongoose.Schema({
  name: {
    type: String,
    required: [true, 'Name is required'],
    trim: true,
  },
  cmsId: {
    type: String,
    required: [true, 'CMS ID is required'],
    unique: true, // Ensuring uniqueness
    trim: true,
  },
  frontImage: {
    type: String, // Store path or URL
    required: [true, 'Front image is required'],
  },
  leftImage: {
    type: String, // Store path or URL
    required: false, // Optional
  },
  rightImage: {
    type: String, // Store path or URL
    required: false, // Optional
  },
  status: {
    type: String,
    enum: ['allowed', 'banned'],
    default: 'allowed',
  },
  lastSeen: {
    type: Date,
    default: Date.now,
  }
}, { timestamps: true });

// Add index for sorting by creation date
PersonSchema.index({ createdAt: -1 });

module.exports = mongoose.model('Person', PersonSchema);