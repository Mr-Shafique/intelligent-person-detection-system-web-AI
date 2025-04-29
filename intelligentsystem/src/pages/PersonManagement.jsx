import { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import Card from '../components/Card';
import Button from '../components/Button';
import InputField from '../components/InputField';
import Modal from '../components/Modal';
import { api } from '../utils/api';

// Import BASE_URL for image paths
const API_URL = 'http://localhost:5000/api';
const BASE_URL = 'http://localhost:5000';

const PersonManagement = () => {
  const [persons, setPersons] = useState([]);
  const [filteredPersons, setFilteredPersons] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedPerson, setSelectedPerson] = useState(null);
  // Keep basic form data separate
  const [formData, setFormData] = useState({
    name: '',
    cmsId: '',
    status: 'allowed',
  });
  // State for image previews (URLs or Base64)
  const [frontImagePreview, setFrontImagePreview] = useState(null);
  const [leftImagePreview, setLeftImagePreview] = useState(null);
  const [rightImagePreview, setRightImagePreview] = useState(null);
  // State for actual image files
  const [frontImageFile, setFrontImageFile] = useState(null);
  const [leftImageFile, setLeftImageFile] = useState(null);
  const [rightImageFile, setRightImageFile] = useState(null);

  useEffect(() => {
    fetchPersons();
  }, []);

  // Add search functionality
  useEffect(() => {
    if (!searchTerm.trim()) {
      // If search term is empty, show all persons
      setFilteredPersons(persons);
    } else {
      // Filter persons by name or CMS ID
      const lowercasedSearch = searchTerm.toLowerCase();
      const filtered = persons.filter(
        person => 
          person.name.toLowerCase().includes(lowercasedSearch) || 
          person.cmsId.toLowerCase().includes(lowercasedSearch)
      );
      setFilteredPersons(filtered);
    }
  }, [searchTerm, persons]);

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  const fetchPersons = async () => {
    try {
      const response = await api.getPersons();
      setPersons(response.data);
      setFilteredPersons(response.data); // Initialize filteredPersons
    } catch (error) {
      toast.error('Error fetching persons: ' + error.message);
    }
  };

  // Generic image handler function
  const handleImageChange = (e, setPreview, setFile) => {
    const file = e.target.files[0];
    if (file) {
      setFile(file); // Store the file object
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result); // Set Base64 preview
      };
      reader.readAsDataURL(file);
    } else {
      // If user cancels file selection, reset preview and file
      setFile(null);
      setPreview(null);
    }
     // Clear the input value to allow selecting the same file again
    e.target.value = null;
  };

  // Specific handlers calling the generic one
  const handleFrontImageChange = (e) => handleImageChange(e, setFrontImagePreview, setFrontImageFile);
  const handleLeftImageChange = (e) => handleImageChange(e, setLeftImagePreview, setLeftImageFile);
  const handleRightImageChange = (e) => handleImageChange(e, setRightImagePreview, setRightImageFile);


  const handleSubmit = async (e) => {
    e.preventDefault();
    
    console.log('Form Submission - Form Data:', formData);
    console.log('Form Submission - Image Files:', {
      frontImageFile: frontImageFile ? `${frontImageFile.name} (${frontImageFile.size} bytes)` : null,
      leftImageFile: leftImageFile ? `${leftImageFile.name} (${leftImageFile.size} bytes)` : null,
      rightImageFile: rightImageFile ? `${rightImageFile.name} (${rightImageFile.size} bytes)` : null
    });
    
    // Validate required fields
    if (!formData.name.trim()) {
      return toast.error('Name is required');
    }
    
    if (!formData.cmsId.trim()) {
      return toast.error('CMS ID is required');
    }

    // Validate front image is present
    if (!selectedPerson && !frontImageFile) { // Only require front image file when adding new
      return toast.error('Front picture is required');
    }
    if (selectedPerson && !frontImagePreview && !frontImageFile) { // Require existing preview or new file when editing
       return toast.error('Front picture is required');
    }

    // Prepare FormData
    const data = new FormData();
    
    // For updates, only include fields that have changed
    if (selectedPerson) {
      console.log('Updating person, checking for changes...');
      
      // Check which text fields have changed
      if (formData.name !== selectedPerson.name) {
        data.append('name', formData.name);
        console.log('Name changed:', selectedPerson.name, '->', formData.name);
      }
      
      if (formData.cmsId !== selectedPerson.cmsId) {
        data.append('cmsId', formData.cmsId);
        console.log('CMS ID changed:', selectedPerson.cmsId, '->', formData.cmsId);
      }
      
      if (formData.status !== selectedPerson.status) {
        data.append('status', formData.status);
        console.log('Status changed:', selectedPerson.status, '->', formData.status);
      }
      
      // Add files only if new ones were selected
      if (frontImageFile) {
        data.append('frontImage', frontImageFile);
        console.log('Front image changed:', frontImageFile.name);
      }
      
      if (leftImageFile) {
        data.append('leftImage', leftImageFile);
        console.log('Left image changed:', leftImageFile.name);
      }
      
      if (rightImageFile) {
        data.append('rightImage', rightImageFile);
        console.log('Right image changed:', rightImageFile.name);
      }
      
      // If no changes were made, alert the user
      if ([...data.entries()].length === 0) {
        toast.info('No changes detected');
        return;
      }
    } else {
      // For new person, include all fields
      data.append('name', formData.name);
      data.append('cmsId', formData.cmsId);
      data.append('status', formData.status);
      
      // Append files
      if (frontImageFile) {
        data.append('frontImage', frontImageFile);
      }
      
      if (leftImageFile) {
        data.append('leftImage', leftImageFile);
      }
      
      if (rightImageFile) {
        data.append('rightImage', rightImageFile);
      }
    }
    
    // Log what's being sent
    console.log('FormData content summary:');
    for (let pair of data.entries()) {
      if (pair[1] instanceof File) {
        console.log(`${pair[0]}: [File: ${pair[1].name}, ${pair[1].size} bytes, ${pair[1].type}]`);
      } else {
        console.log(`${pair[0]}: ${pair[1]}`);
      }
    }

    try {
      if (selectedPerson) {
        // Update existing person
        await api.updatePerson(selectedPerson._id, data);
        toast.success('Person updated successfully');
      } else {
        // Add new person
        await api.addPerson(data);
        toast.success('Person added successfully');
      }
      setIsModalOpen(false);
      fetchPersons(); // Refresh the list
    } catch (error) {
      // Handle errors
      toast.error('Error saving person: ' + (error.response?.data?.message || error.message));
    }
  };

  const handleDelete = async (id) => {
    if (window.confirm('Are you sure you want to delete this person?')) {
      try {
        await api.deletePerson(id);
        toast.success('Person deleted successfully');
        fetchPersons();
      } catch (error) {
        toast.error('Error deleting person: ' + error.message);
      }
    }
  };

  const openModal = (person = null) => {
    setSelectedPerson(person);
    if (person) {
      // Editing: Populate form data and image previews from person object
      setFormData({
        name: person.name || '',
        cmsId: person.cmsId || '',
        status: person.status || 'allowed',
      });
      // Assuming backend returns image URLs in person.frontImage, etc.
      setFrontImagePreview(person.frontImage || null);
      setLeftImagePreview(person.leftImage || null);
      setRightImagePreview(person.rightImage || null);
    } else {
      // Adding: Reset form data and previews
      setFormData({
        name: '',
        cmsId: '',
        status: 'allowed',
      });
      setFrontImagePreview(null);
      setLeftImagePreview(null);
      setRightImagePreview(null);
    }
    // Reset file states for both add and edit
    setFrontImageFile(null);
    setLeftImageFile(null);
    setRightImageFile(null);
    setIsModalOpen(true);
  };

  return (
    <div className="space-y-6">
      <div className="flex  gap-4 justify-around  flex-column  py-4">
      {/* Search Row */}
        <div className="relative flex flex-1">
          <div className="absolute inset-y-0 pl-2 left-0 flex items-center  pointer-events-none">
            <svg className="w-4 h-4 text-gray-500" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 20">
              <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z"/>
            </svg>
          </div>
          <input
            type="search"
            className="block w-full p-3 pl-10 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-blue-500 focus:border-blue-500"
            placeholder="Search by name or CMS ID..."
            value={searchTerm}
            onChange={handleSearchChange}
            />
        </div>
        {searchTerm && (
          <p className="mt-2 text-sm text-gray-500">
            Found {filteredPersons.length} {filteredPersons.length === 1 ? 'result' : 'results'} for "{searchTerm}"
          </p>
        )}
        <div className='flex flex-2'>
        <Button onClick={() => openModal()}>Add New Person</Button>
        </div>
        </div>

      <Card className="p-4">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Image
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  CMS ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredPersons.map((person) => (
                <tr key={person._id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {person.frontImage ? (
                      <img
                        src={`${BASE_URL}${person.frontImage}`}
                        alt={`${person.name} (Front)`}
                        className="h-16 w-16 rounded-full object-cover border-2 border-gray-200"
                        onError={(e) => {
                          console.error("Image load error:", e.target.src);
                          e.target.onerror = null; // Prevent infinite error loop
                          e.target.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23999' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'%3E%3C/path%3E%3Ccircle cx='12' cy='7' r='4'%3E%3C/circle%3E%3C/svg%3E";
                        }}
                      />
                    ) : (
                      <div className="h-16 w-16 rounded-full flex items-center justify-center bg-gray-100 border-2 border-gray-200 text-gray-500 text-xs">
                        None
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">
                      {person.name}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-500">
                      {person.cmsId}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        person.status === 'allowed'
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}
                    >
                      {person.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button
                      onClick={() => openModal(person)}
                      className="text-indigo-600 hover:text-indigo-900 mr-4"
                    >
                      Edit
                    </button>
                    <button
                      onClick={() => handleDelete(person._id)}
                      className="text-red-600 hover:text-red-900"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Modal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)}>
        <form onSubmit={handleSubmit} className="space-y-4">
          <h2 className="text-xl font-semibold mb-4">
            {selectedPerson ? 'Edit Person' : 'Add New Person'}
          </h2>
          <InputField
            label="Name"
            value={formData.name}
            onChange={(e) =>
              setFormData({ ...formData, name: e.target.value })
            }
            required
          />
          <InputField
            label="CMS ID"
            value={formData.cmsId}
            onChange={(e) =>
              setFormData({ ...formData, cmsId: e.target.value })
            }
            required
          />
          {/* --- New Image Input Sections --- */}
          {/* Front Image Input */}
          <div className="space-y-1">
            <label className="block text-sm font-medium text-gray-700">
              Picture <span className="text-red-500">*</span>
            </label>
            <div className="flex items-center space-x-4">
              {frontImagePreview ? (
                <img
                  src={frontImagePreview.startsWith('/') ? `${BASE_URL}${frontImagePreview}` : frontImagePreview}
                  alt="Front Preview"
                  className="h-16 w-16 rounded-full object-cover border-2 border-gray-200"
                />
              ) : (
                <div className="h-16 w-16 rounded-full flex items-center justify-center bg-gray-100 border-2 border-gray-200 text-gray-500 text-xs">
                  None
                </div>
              )}
              <input
                type="file"
                accept="image/*"
                onChange={handleFrontImageChange}
                className="file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
              />
            </div>
          </div>

          {/* Left Image Input */}
          <div className="space-y-1">
            <label className="block text-sm font-medium text-gray-700">
              Picture
            </label>
            <div className="flex items-center space-x-4">
              {leftImagePreview ? (
                <img
                  src={leftImagePreview.startsWith('/') ? `${BASE_URL}${leftImagePreview}` : leftImagePreview}
                  alt="Left Preview"
                  className="h-16 w-16 rounded-full object-cover border-2 border-gray-200"
                />
              ) : (
                <div className="h-16 w-16 rounded-full flex items-center justify-center bg-gray-100 border-2 border-gray-200 text-gray-500 text-xs">
                  <img 
                  src="https://cdn.pixabay.com/photo/2015/10/05/22/37/blank-profile-picture-973460_1280.png" 
                  alt="person"
                  className=" rounded-full object-center" 
                   />
                </div>
              )}
              <input
                type="file"
                accept="image/*"
                onChange={handleLeftImageChange}
                className="file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
              />
            </div>
          </div>

          {/* Right Image Input */}
          <div className="space-y-1">
            <label className="block text-sm font-medium text-gray-700">
              Picture
            </label>
            <div className="flex items-center space-x-4">
              {rightImagePreview ? (
                <img
                  src={rightImagePreview.startsWith('/') ? `${BASE_URL}${rightImagePreview}` : rightImagePreview}
                  alt="Right Preview" 
                  className="h-16 w-16 rounded-full object-cover border-2 border-gray-200"
                />
              ) : (
                <div className="h-16 w-16 rounded-full flex items-center justify-center bg-gray-100 border-2 border-gray-200 text-gray-500 text-xs">
                  None
                </div>
              )}
              <input
                type="file"
                accept="image/*"
                onChange={handleRightImageChange}
                className="file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
              />
            </div>
          </div>
          {/* --- End New Image Input Sections --- */}

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Status
            </label>
            <select
              value={formData.status}
              onChange={(e) =>
                setFormData({ ...formData, status: e.target.value })
              }
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            >
              <option value="allowed">Allowed</option>
              <option value="banned">Banned</option>
            </select>
          </div>
          <div className="flex justify-end space-x-4">
            <Button
              type="button"
              onClick={() => setIsModalOpen(false)}
              className="bg-gray-500 hover:bg-gray-600"
            >
              Cancel
            </Button>
            <Button type="submit">
              {selectedPerson ? 'Update' : 'Add'} Person
            </Button>
          </div>
        </form>
      </Modal>
    </div>
  );
};

export default PersonManagement;