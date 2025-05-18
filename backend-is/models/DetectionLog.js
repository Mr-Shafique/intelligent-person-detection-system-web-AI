const mongoose = require('mongoose');


const EventSchema = new mongoose.Schema({
  timestamp: {
    type: Date,
    required: true
  },
  action: {
    type: String,
    enum: ['IN', 'OUT'],
    required: true
  },
  camera_source: {
    type: String,
    required: true
  },
  image_saved: {
    type: String,
    required: false
  },
  person_status_at_event: { // Added field to store status at the time of event
    type: String,
    required: true, // Changed to true
    default: 'Unknown' // Added default
  }
}, { _id: false });

const DetectionLogSchema = new mongoose.Schema({
  person_cmsId: {
    type: String,
    required: true,
    index: true
  },
  person_name: {
    type: String,
    required: true
  },
  status: { // Add the status field
    type: String,
    required: true, // Changed to true
    default: 'Unknown' // Added default
  },
  events: {
    type: [EventSchema],
    default: []
  }
}, { timestamps: true });

DetectionLogSchema.index({ person_cmsId: 1 });

module.exports = mongoose.model('DetectionLog', DetectionLogSchema);