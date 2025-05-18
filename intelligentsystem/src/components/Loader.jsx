
import React from 'react';
import { Atom } from 'react-loading-indicators';


const Loader = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full w-full">
      <Atom color="#0000FF" size="medium" text="" textColor="" />
    </div>
  );
};

export default Loader;