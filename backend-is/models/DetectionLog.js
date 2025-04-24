const mongoose = require('mongoose');

const DetectionLogSchema = new mongoose.Schema({
  person: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Person', // Reference to the Person model
    required: true
  },
  status: {
    type: String,
    enum: ['allowed', 'banned'],
    required: true // Status at the time of detection
  },
  location: {
    type: String,
    required: true,
    trim: true
  },
  capturedImage: {
    type: String, // Store image as base64 string or URL
    required: true
  },
  timestamp: {
    type: Date,
    default: Date.now
  }
}, { timestamps: true }); // Add createdAt and updatedAt automatically

// Index for faster querying by timestamp
DetectionLogSchema.index({ timestamp: -1 });

module.exports = mongoose.model('DetectionLog', DetectionLogSchema);