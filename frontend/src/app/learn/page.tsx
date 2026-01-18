'use client';

import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { useMutation } from "@tanstack/react-query";
import { useGlobalContext } from "@/src/context/GlobalContext";
import { predictionApi } from "@/src/lib/api/prediction";
import { Loader2, Camera, ChevronLeft, ChevronRight, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import HandModelViewer from "@/src/components/learn/HandModelViewer";
import { courseApi } from "@/src/lib/api/course";

// Standard MediaPipe Hand Connections
const LANDMARK_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // Index
  [5, 9], [9, 10], [10, 11], [11, 12], // Middle
  [9, 13], [13, 14], [14, 15], [15, 16], // Ring
  [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [0, 17], // Palm
];

export default function LearnPage() {
  const { originalNepaliAlphabets, originalNslLetters } = useGlobalContext();
  
  // State
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [characterIndex, setCharacterIndex] = useState(0);
  const [predictionResult, setPredictionResult] = useState({ prediction: "--", confidence: 0 });
  const [isCorrect, setIsCorrect] = useState(false);
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const requestRef = useRef<number | null>(null);
  const lastApiCallTimeRef = useRef<number>(0);
  
  // --- NEW: Ref to lock predictions instantly without waiting for re-renders ---
  const isPausedRef = useRef(false);
  
  const { mutate: trackProgress } = useMutation({
    mutationFn: courseApi.incrementProgress,
    onError: (err) => console.error("Failed to track progress:", err),
  });
  // --- API MUTATION ---
  const { mutate: predictSign } = useMutation({
    mutationFn: predictionApi.predict,
    onSuccess: (data) => {
      setPredictionResult(data);
      
      const targetChar = originalNslLetters[characterIndex];
      
      // Check if Correct AND not already paused
      if (data.prediction === targetChar && data.confidence >= 0.95 && !isPausedRef.current) {
        // 1. LOCK IMMEDIATELY
        isPausedRef.current = true;
        
        setIsCorrect(true);
         trackProgress(targetChar); 
        
        setTimeout(() => {
          handleNextChar(); // Move to next
          setIsCorrect(false); 
          // 2. UNLOCK after transition is done
          isPausedRef.current = false; 
        }, 2000);
      }
    },
    onError: (err) => console.error("Prediction Error:", err)
  });

  // --- LOGIC: Normalize Landmarks ---
  const normalizeLandmarks = (landmarks: any[]) => {
    const landmarkArray = landmarks.map((lm) => [lm.x, lm.y, lm.z]);
    const center = landmarkArray.reduce((acc, [x, y, z]) => [acc[0] + x, acc[1] + y, acc[2] + z], [0, 0, 0]).map((sum) => sum / landmarkArray.length);
    const normalized = landmarkArray.map(([x, y, z]) => [x - center[0], y - center[1], z - center[2]]);
    const maxDistance = Math.max(...normalized.map(([x, y, z]) => Math.sqrt(x * x + y * y + z * z)));
    return normalized.map(([x, y, z]) => [x / maxDistance, y / maxDistance, z / maxDistance]);
  };

  // --- SETUP: Initialize MediaPipe ---
  useEffect(() => {
    const initMediaPipe = async () => {
      try {
        const visionFileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(visionFileset, {
          baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/hand_landmarker.task", delegate: "GPU" },
          runningMode: "VIDEO",
          numHands: 1,
        });
        setIsModelLoaded(true);
      } catch (error) {
        console.error("Error loading MediaPipe:", error);
      }
    };
    initMediaPipe();
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      handLandmarkerRef.current?.close();
    };
  }, []);

  // --- LOOP: Prediction Loop ---
  const processVideo = () => {
    if (!videoRef.current || !canvasRef.current || !handLandmarkerRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const results = handLandmarkerRef.current.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (results.landmarks) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-canvas.width, 0);
      results.landmarks.forEach((landmarks) => {
        LANDMARK_CONNECTIONS.forEach(([start, end]) => {
          const s = landmarks[start];
          const e = landmarks[end];
          ctx.beginPath();
          ctx.moveTo(s.x * canvas.width, s.y * canvas.height);
          ctx.lineTo(e.x * canvas.width, e.y * canvas.height);
          ctx.strokeStyle = "#22c55e";
          ctx.lineWidth = 3;
          ctx.stroke();
        });
        landmarks.forEach((lm) => {
          ctx.beginPath();
          ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 5, 0, 2 * Math.PI);
          ctx.fillStyle = "#ef4444";
          ctx.fill();
        });

        const now = performance.now();
        // --- CHECK LOCK: Don't send if paused ---
        if (!isPausedRef.current && now - lastApiCallTimeRef.current >= 1000) {
          lastApiCallTimeRef.current = now;
          predictSign(normalizeLandmarks(landmarks));
        }
      });
      ctx.restore();
    }
    requestRef.current = requestAnimationFrame(processVideo);
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.addEventListener("loadeddata", processVideo);
        setCameraActive(true);
      }
    } catch (err) { console.error("Camera Error:", err); }
  };
  const stopCamera = () => {
    // 1. Stop the animation loop
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
      requestRef.current = null;
    }

    // 2. Stop the video stream tracks (releases hardware)
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    // 3. Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    // 4. Update UI state
    setCameraActive(false);
  };


  const handleNextChar = () => setCharacterIndex((prev) => (prev + 1) % originalNepaliAlphabets.length);
  const handlePrevChar = () => setCharacterIndex((prev) => (prev - 1 + originalNepaliAlphabets.length) % originalNepaliAlphabets.length);

  return (
    // MAIN LAYOUT: 2 Columns
    // Left: Flex Col (Practice + 3D)
    // Right: Full Height Camera
    <div className="h-full w-full grid grid-cols-1 lg:grid-cols-2 gap-4 p-2 overflow-hidden">
      
      {/* --- LEFT COLUMN (Practice & 3D) --- */}
      <div className="flex flex-col gap-4 h-full overflow-hidden">
        
        {/* ROW 1: LESSON CARD */}
        <div className="flex-1 rounded-3xl overflow-hidden shadow-sm relative">
           <HandModelViewer character={originalNslLetters[characterIndex]} />
        </div>
        

        {/* ROW 2: 3D VIEWER */}
        <div className="flex-1 flex flex-col bg-card dark:bg-card rounded-3xl p-6 shadow-sm border border-border relative overflow-hidden">
          
          {/* Header */}
          <div className="flex justify-between items-start z-10 shrink-0">
            <div>
              <h1 className="text-sm md:text-2xl font-semibold md:font-bold">Practice Mode</h1>
              <p className="text-xs md:text-sm text-muted-foreground">Replicate this sign</p>
            </div>
            <div className="flex flex-col items-end">
              <div className="px-3 py-1 bg-primary/10 text-primary rounded-full text-[0.65rem] md:text-xs font-semibold md:font-bold uppercase tracking-wider mb-1">
                Target: {originalNslLetters[characterIndex]}
              </div>
              <span className="text-[10px] pr-3 font-medium text-muted-foreground">
                {characterIndex + 1} / {originalNepaliAlphabets.length}
              </span>
            </div>
          </div>

          {/* Middle: Carousel */}
          <div className="flex-1 flex items-center justify-between z-10 relative px-2">
            <button onClick={handlePrevChar} className="p-3 rounded-full bg-secondary/30 hover:bg-secondary text-muted-foreground hover:text-foreground transition-all">
              <ChevronLeft size={28} />
            </button>

            <div className="relative flex items-center justify-center">
              <div className={cn(
                "text-[3rem] lg:text-[7rem] font-bold leading-none transition-all duration-500 select-none pb-4",
                isCorrect ? "text-green-500 scale-110 drop-shadow-lg" : "text-foreground"
              )}>
                {originalNepaliAlphabets[characterIndex]}
              </div>
              {isCorrect && (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                  <CheckCircle2 size={120} className="text-green-500/20 animate-ping" />
                </div>
              )}
            </div>

            <button onClick={handleNextChar} className="p-3 rounded-full bg-secondary/30 hover:bg-secondary text-muted-foreground hover:text-foreground transition-all">
              <ChevronRight size={28} />
            </button>
          </div>

          {/* Footer: Stats */}
          <div className="z-10 shrink-0">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-secondary p-3 rounded-2xl flex flex-col items-center">
                <span className="text-[10px] uppercase font-bold text-muted-foreground">Prediction</span>
                <span className={cn("text-xl font-bold", predictionResult.prediction === originalNslLetters[characterIndex] ? "text-green-500" : "text-foreground")}>
                  {predictionResult.prediction}
                </span>
              </div>
              <div className="bg-secondary p-3 rounded-2xl flex flex-col items-center">
                <span className="text-[10px] uppercase font-bold text-muted-foreground">Confidence</span>
                <span className="text-xl font-bold text-blue-500">{(predictionResult.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* --- RIGHT COLUMN (Camera) --- */}
      <div className="h-full min-h-60 bg-black rounded-3xl overflow-hidden relative shadow-lg flex items-center justify-center group ">
        {!isModelLoaded && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white bg-neutral-900 z-30">
            <Loader2 className="w-8 h-8 animate-spin text-blue-500 mb-2" />
            <p className="text-xs font-medium">Loading Vision...</p>
          </div>
        )}
        {isModelLoaded && !cameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white bg-neutral-900/80 backdrop-blur-sm z-30">
             <button onClick={startCamera} className="flex flex-col items-center gap-3 group-hover:scale-105 transition-transform">
               <div className="w-20 h-20 bg-blue-600 rounded-full flex items-center justify-center shadow-lg shadow-blue-600/30">
                 <Camera size={32} />
               </div>
               <span className="text-xl font-bold">Start Camera</span>
             </button>
          </div>
        )}
        <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover opacity-80" playsInline muted autoPlay style={{ transform: "scaleX(-1)" }} />
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover z-20" />
        {cameraActive && (
          <>
            {/* Live Indicator (Top Right) */}
            <div className="absolute top-4 right-4 z-30 bg-black/60 px-3 py-1.5 rounded-full flex items-center gap-2 border border-white/10 shadow-xl pointer-events-none">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              <span className="text-white text-xs font-bold uppercase tracking-wider">Live</span>
            </div>

            {/* Stop Button (Bottom Center) */}
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-30">
              <button 
                onClick={stopCamera}
                className="flex items-center gap-2 bg-red-500/90 hover:bg-red-600 text-white px-6 py-2.5 rounded-full font-bold shadow-lg transition-all hover:scale-105 active:scale-95 backdrop-blur-sm"
              >
                <div className="w-3 h-3 bg-white rounded-[2px]" /> {/* Square Stop Icon */}
                <span>Stop Camera</span>
              </button>
            </div>
          </>
        )}
      </div>

    </div>
  );
}