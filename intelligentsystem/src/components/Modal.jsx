import React, { useEffect } from 'react';

const Modal = ({ isOpen, onClose, title, children }) => {
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{
        backdropFilter: 'blur(8px)', // background blur
        backgroundColor: 'rgba(0,0,0,0.3)', // semi-transparent dark overlay
      }}
    >
      <div
        className="bg-white rounded-lg shadow-lg relative"
        style={{
          margin: 20, // 20px margin around the popup
          maxHeight: '90vh',
          overflowY: 'auto',
          minWidth: 320,
        }}
      >
        <button
          onClick={onClose}
          className="absolute top-2 right-2 text-gray-500 hover:text-gray-700 text-xl font-bold"
        >
          &times;
        </button>
        {title && <div className="px-6 pt-6 pb-2 text-lg font-semibold">{title}</div>}
        <div className="px-6 pb-6">{children}</div>
      </div>
    </div>
  );
};

export default Modal;