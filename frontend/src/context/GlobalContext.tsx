'use client';

import { createContext, useContext, useState, ReactNode } from 'react';

// --- CONSTANTS ---
const INITIAL_NEPALI = [
  "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ",
  "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न",
  "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श",
  "ष", "स", "ह", "क्ष", "त्र", "ज्ञ",
];

const INITIAL_NSL = [
  "Ka", "Kha", "Ga", "Gha", "Nga", "Cha", "Chha", "Ja", "Jha", "Yan",
  "Ta", "Tha", "Da", "Dha", "Na", "Taa", "Thaa", "Daa", "Dhaa", "Naa",
  "Pa", "Pha", "Ba", "Bha", "Ma", "Ya", "Ra", "La", "Wa", "T_Sha",
  "M_Sha", "D_Sha", "Ha", "Ksha", "Tra", "Gya",
];

type GlobalContextType = {
  isSignUpMode: boolean;
  setIsSignUpMode: (isSignUpMode: boolean) => void;
  originalNepaliAlphabets: string[];
  originalNslLetters: string[];
  getCookie: (name: string) => string | null;
};

const GlobalContext = createContext<GlobalContextType | undefined>(undefined);

export function GlobalProvider({ children }: { children: ReactNode }) {
  const [isSignUpMode, setIsSignUpMode] = useState(false);
  
  // We keep these in state in case you want to modify/filter them later
  const [originalNepaliAlphabets] = useState(INITIAL_NEPALI);
  const [originalNslLetters] = useState(INITIAL_NSL);

  // Helper function from your legacy code
  const getCookie = (name: string) => {
    if (typeof document === 'undefined') return null; // Server-side safety
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
      const cookies = document.cookie.split(";");
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.startsWith(name + "=")) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  };

  return (
    <GlobalContext.Provider value={{ 
      isSignUpMode, 
      setIsSignUpMode,
      originalNepaliAlphabets,
      originalNslLetters,
      getCookie
    }}>
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