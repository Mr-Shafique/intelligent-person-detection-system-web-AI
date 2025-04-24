import axios from 'axios';
import { mockDetectionLogs } from './mockData';

const API_URL = 'http://localhost:5000/api';

// Simulate API delay for mock data
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export const api = {
  // Persons API - Real MongoDB Connection
  getPersons: async () => {
    try {
      const response = await axios.get(`${API_URL}/persons`);
      return { data: response.data };
    } catch (error) {
      console.error('Error fetching persons:', error);
      throw error;
    }
  },

  addPerson: async (person) => {
    try {
      const response = await axios.post(`${API_URL}/persons`, person);
      return { data: response.data };
    } catch (error) {
      console.error('Error adding person:', error);
      throw error;
    }
  },

  updatePerson: async (id, updates) => {
    try {
      const response = await axios.put(`${API_URL}/persons/${id}`, updates);
      return { data: response.data };
    } catch (error) {
      console.error('Error updating person:', error);
      throw error;
    }
  },

  deletePerson: async (id) => {
    try {
      await axios.delete(`${API_URL}/persons/${id}`);
      return { success: true };
    } catch (error) {
      console.error('Error deleting person:', error);
      throw error;
    }
  },

  // Detection Logs API - Real MongoDB Connection
  getDetectionLogs: async () => {
    try {
      const response = await axios.get(`${API_URL}/detections`);
      return { data: response.data };
    } catch (error) {
      console.error('Error fetching detection logs:', error);
      throw error;
    }
  },

  // POST endpoint is primarily for the camera system, but we keep the function signature
  // If needed for testing, you could implement a way to call this from the frontend
  addDetectionLog: async (log) => {
    try {
      // Example: This assumes log contains { personId, location, capturedImage }
      const response = await axios.post(`${API_URL}/detections`, log);
      return { data: response.data };
    } catch (error) {
      console.error('Error adding detection log:', error);
      throw error;
    }
  },

  // Settings API
  updateSettings: async (settings) => {
    await delay(500);
    localStorage.setItem('settings', JSON.stringify(settings));
    return { data: settings };
  },

  getSettings: async () => {
    await delay(500);
    const settings = localStorage.getItem('settings');
    return { data: settings ? JSON.parse(settings) : {} };
  },
};