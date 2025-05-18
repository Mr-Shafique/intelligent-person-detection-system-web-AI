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
    <div className="fixed inset-0 z-50 flex items-center justify-center   ">
      {/* Fullscreen blurred overlay */}
      <div
        className="absolute inset-0 w-full h-full"
        style={{
          backdropFilter: 'blur(10px)',
          WebkitBackdropFilter: 'blur(10px)',
          backgroundColor: 'rgba(30,30,30,0.35)',
          transition: 'background 0.3s',
          zIndex: 0,
        }}
      />
      {/* Modal content */}
      <div
        className="bg-white rounded-2xl shadow-2xl relative w-full max-w-2xl mx-auto animate-fade-in"
        style={{
          margin: 30,
          maxHeight: '90vh',
          overflowY: 'auto',
          boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
          border: '1px solid rgba(255,255,255,0.18)',
          zIndex: 10,
        }}
      >
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-red-500 text-3xl font-bold transition"
          aria-label="Close"
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            zIndex: 20,
          }}
        >
          &times;
        </button>
        {title && (
          <div className="px-8 pt-8 pb-2 text-2xl font-bold text-gray-800 border-b border-gray-100">
            {title}
          </div>
        )}
        <div className="px-8 py-6">{children}</div>
      </div>
      <style>
        {`
          @media (max-width: 600px) {
            .max-w-2xl { max-width: 98vw !important; }
            .px-8 { padding-left: 1rem !important; padding-right: 1rem !important; }
          }
          .animate-fade-in {
            animation: fadeInModal 0.25s;
          }
          @keyframes fadeInModal {
            from { opacity: 0; transform: translateY(30px);}
            to { opacity: 1; transform: translateY(0);}
          }
        `}
      </style>
    </div>
  );
};

export default Modal;