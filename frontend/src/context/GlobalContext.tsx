'use client';

import { createContext, useContext, useState, ReactNode } from 'react';

type GlobalContextType = {
  isSignUpMode: boolean;
  setIsSignUpMode: (isSignUpMode: boolean) => void;
};

const GlobalContext = createContext<GlobalContextType | undefined>(undefined);

export function GlobalProvider({ children }: { children: ReactNode }) {
    const [isSignUpMode, setIsSignUpMode] = useState(false);

  return (
    <GlobalContext.Provider value={{ isSignUpMode, setIsSignUpMode }}>
      {children}
    </GlobalContext.Provider>
  );
}

export function useGlobalContext() {
  const context = useContext(GlobalContext);
  if (!context) {
    throw new Error('useGlobalContext must be used within a GlobalProvider');
  }
  return context;
}
