'use client';

import { useState } from 'react';
import Image from 'next/image';
import { User, Lock, Mail, Loader2 } from 'lucide-react';
import { useAuth } from '@/src/hooks/useAuth';
import styles from '@/src/components/auth/auth.module.css';
import clsx from 'clsx';
import log from '@/src/assets/images/log.svg';
import reg from '@/src/assets/images/register.svg';
import { useGlobalContext } from '@/src/context/GlobalContext';

export default function AuthPage() {
  const {isSignUpMode, setIsSignUpMode} = useGlobalContext();
  
  const { 
    login, isLoggingIn, loginError, 
    register, isRegistering, registerError 
  } = useAuth();

  const [loginData, setLoginData] = useState({ username: '', password: '' });
  const [registerData, setRegisterData] = useState({ username: '', email: '', password: '' });

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    login(loginData);
  };

  const handleRegister = (e: React.FormEvent) => {
    e.preventDefault();
    register(registerData);
  };

  return (
    <div className={clsx(
      "relative h-full w-full bg-white overflow-hidden",
      styles.container,
      isSignUpMode && styles.signUpMode
    )}>
      
      {/* Background Circle */}
      <div className={styles.circle} />

      {/* Forms Container */}
      <div className="absolute w-full h-full top-0 left-0">
        <div className={styles.signinSignup}>
          
          {/* LOGIN FORM */}
          <form 
            onSubmit={handleLogin} 
            className={clsx(
              "flex items-center justify-center flex-col overflow-hidden transition-all duration-[200ms] delay-[700ms] ease-in-out col-start-1 row-start-1 z-[2]",
              "px-5 md:px-10",
              isSignUpMode ? "opacity-0 z-[1]" : "opacity-100 z-[2]"
            )}
          >
            <h2 className="text-3xl md:text-4xl text-neutral-700 font-bold mb-3 md:mb-5">Sign in</h2>
            
            <div className="w-full max-w-[380px] bg-gray-100 rounded-full h-11 md:h-[55px] my-2 grid grid-cols-[15%_85%] px-4 relative items-center">
              <div className="flex items-center justify-center text-neutral-400"><User size={20} /></div>
              <input 
                type="text" placeholder="Username" required value={loginData.username}
                onChange={(e) => setLoginData({...loginData, username: e.target.value})}
                className="bg-transparent border-none outline-none text-base md:text-lg font-semibold text-neutral-700 placeholder:text-neutral-400 w-full"
              />
            </div>
            
            <div className="w-full max-w-[380px] bg-gray-100 rounded-full h-11 md:h-[55px] my-2 grid grid-cols-[15%_85%] px-4 relative items-center">
              <div className="flex items-center justify-center text-neutral-400"><Lock size={20} /></div>
              <input 
                type="password" placeholder="Password" required value={loginData.password}
                onChange={(e) => setLoginData({...loginData, password: e.target.value})}
                className="bg-transparent border-none outline-none text-base md:text-lg font-semibold text-neutral-700 placeholder:text-neutral-400 w-full"
              />
            </div>

            {loginError && (
              <p className="text-red-500 text-xs md:text-sm mt-2 font-medium max-w-75 text-center">
                {(loginError as any)?.response?.data?.detail || "Invalid credentials"}
              </p>
            )}

            <button type="submit" disabled={isLoggingIn} className="w-32 md:w-[150px] h-10 md:h-[49px] bg-blue-500 hover:bg-blue-600 border-none rounded-full cursor-pointer text-white uppercase font-bold text-xs md:text-sm my-3 md:my-5 transition-colors duration-300 flex items-center justify-center shadow-md">
               {isLoggingIn ? <Loader2 className="animate-spin" /> : "Login"}
            </button>
            
            
          </form>

          {/* REGISTER FORM */}
          <form 
            onSubmit={handleRegister} 
            className={clsx(
              "flex items-center justify-center flex-col overflow-hidden transition-all duration-[200ms] delay-[700ms] ease-in-out col-start-1 row-start-1",
              "px-5 md:px-10",
              isSignUpMode ? "opacity-100 z-[2]" : "opacity-0 z-[1]"
            )}
          >
            <h2 className="text-3xl md:text-4xl text-neutral-700 font-bold mb-3 md:mb-5">Sign up</h2>
            
            <div className="w-full max-w-[380px] bg-gray-100 rounded-full h-11 md:h-[55px] my-2 grid grid-cols-[15%_85%] px-4 relative items-center">
              <div className="flex items-center justify-center text-neutral-400"><User size={20} /></div>
              <input 
                type="text" placeholder="Username" required value={registerData.username}
                onChange={(e) => setRegisterData({...registerData, username: e.target.value})}
                className="bg-transparent border-none outline-none text-base md:text-lg font-semibold text-neutral-700 placeholder:text-neutral-400 w-full"
              />
            </div>
            
            <div className="w-full max-w-[380px] bg-gray-100 rounded-full h-11 md:h-[55px] my-2 grid grid-cols-[15%_85%] px-4 relative items-center">
              <div className="flex items-center justify-center text-neutral-400"><Mail size={20} /></div>
              <input 
                type="email" placeholder="Email" required value={registerData.email}
                onChange={(e) => setRegisterData({...registerData, email: e.target.value})}
                className="bg-transparent border-none outline-none text-base md:text-lg font-semibold text-neutral-700 placeholder:text-neutral-400 w-full"
              />
            </div>
            
            <div className="w-full max-w-[380px] bg-gray-100 rounded-full h-11 md:h-[55px] my-2 grid grid-cols-[15%_85%] px-4 relative items-center">
              <div className="flex items-center justify-center text-neutral-400"><Lock size={20} /></div>
              <input 
                type="password" placeholder="Password" required value={registerData.password}
                onChange={(e) => setRegisterData({...registerData, password: e.target.value})}
                className="bg-transparent border-none outline-none text-base md:text-lg font-semibold text-neutral-700 placeholder:text-neutral-400 w-full"
              />
            </div>

            {registerError && (
              <p className="text-red-500 text-xs md:text-sm mt-2 font-medium max-w-75 text-center">
                {(registerError as any)?.response?.data?.detail || "Registration failed"}
              </p>
            )}

            <button type="submit" disabled={isRegistering} className="w-32 md:w-[150px] h-10 md:h-[49px] bg-blue-500 hover:bg-blue-600 border-none rounded-full cursor-pointer text-white uppercase font-bold text-xs md:text-sm my-3 md:my-5 transition-colors duration-300 flex items-center justify-center shadow-md">
               {isRegistering ? <Loader2 className="animate-spin" /> : "Sign Up"}
            </button>
            
            
          </form>
        </div>
      </div>

      {/* --- PANELS CONTAINER --- */}
      {/* On mobile: absolute positioning. On desktop: grid */}
      <div className="absolute w-full h-full top-0 left-0 md:grid md:grid-cols-2">
        
        {/* LEFT PANEL (New Here? -> Sign Up) */}
        <div className={clsx(
          "flex flex-col items-center justify-center text-center z-[7]",
          // Desktop positioning
          "md:items-end md:pr-[17%] md:pl-[12%] md:pt-12 md:pb-8",
          // Mobile positioning: Absolute Top
          "absolute top-0 left-0 w-full p-[2.5rem_8%] md:static md:w-auto md:p-0",
          isSignUpMode ? "pointer-events-none" : "pointer-events-auto",
          styles.leftPanel
        )}>
          <div className={clsx(styles.panelContent, "text-white")}>
            <h3 className="font-bold text-xl md:text-2xl">New here?</h3>
            <p className="text-sm py-3 md:py-4">Join our platform to start learning Nepali Sign Language today.</p>
            <button 
              onClick={() => setIsSignUpMode(true)}
              className="w-[110px] md:w-[130px] h-9 md:h-[41px] border-2 border-white bg-transparent text-white uppercase font-semibold text-xs rounded-full hover:bg-white hover:text-blue-500 transition-colors cursor-pointer"
            >
              Sign up
            </button>
          </div>
          <div className={clsx(styles.panelImage, "w-[200px] md:w-full mt-4")}>
            <Image src={log} alt="Login" width={500} height={500} priority className="w-full h-auto" />
          </div>
        </div>

        {/* RIGHT PANEL (One of us? -> Sign In) */}
        <div className={clsx(
          "flex flex-col items-center justify-center text-center z-[7]",
          // Desktop positioning
          "md:items-start md:pl-[17%] md:pr-[12%] md:pt-12 md:pb-8",
          // Mobile positioning: Absolute Bottom
          "absolute bottom-0 left-0 w-full p-[2.5rem_8%] md:static md:w-auto md:p-0",
          isSignUpMode ? "pointer-events-auto" : "pointer-events-none",
          styles.rightPanel
        )}>
          <div className={clsx(styles.panelContent, "text-white")}>
            <h3 className="font-bold text-xl md:text-2xl">One of us?</h3>
            <p className="text-sm py-3 md:py-4">Already have an account? Sign in to track your progress.</p>
            <button 
              onClick={() => setIsSignUpMode(false)}
              className="w-[110px] md:w-[130px] h-9 md:h-[41px] border-2 border-white bg-transparent text-white uppercase font-semibold text-xs rounded-full hover:bg-white hover:text-blue-500 transition-colors cursor-pointer"
            >
              Sign in
            </button>
          </div>
          <div className={clsx(styles.panelImage, "w-50  absolute md:static -z-10 bottom-40 right-2 md:w-full mt-4")}>
            <Image src={reg} alt="Register" width={500} height={500} className="w-full h-auto" />
          </div>
        </div>

      </div>
    </div>
  );
}