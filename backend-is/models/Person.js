const mongoose = require('mongoose');

const PersonSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true
  },
  cmsId: {
    type: String,
    required: true,
    trim: true,
    unique: true
  },
  image: {
    type: String,
    default: 'https://cdn.pixabay.com/photo/2015/10/05/22/37/blank-profile-picture-973460_1280.png'
  },
  status: {
    type: String,
    enum: ['allowed', 'banned'],
    default: 'allowed'
  },
  lastSeen: {
    type: Date,
    default: Date.now
  }
}, { timestamps: true });

// Add index for sorting by creation date
PersonSchema.index({ createdAt: -1 });

module.exports = mongoose.model('Person', PersonSchema);