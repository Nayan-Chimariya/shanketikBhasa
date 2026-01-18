'use client';

import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { useMutation } from "@tanstack/react-query";
import { useGlobalContext } from "@/src/context/GlobalContext";
import { predictionApi } from "@/src/lib/api/prediction";
import { Loader2, Camera, Play, RotateCcw, Trophy, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

// --- CONFIGURATION ---
const QUESTIONS_PER_TEST = 10;

const LANDMARK_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // Index
  [5, 9], [9, 10], [10, 11], [11, 12], // Middle
  [9, 13], [13, 14], [14, 15], [15, 16], // Ring
  [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [0, 17], // Palm
];

type TestState = 'idle' | 'running' | 'completed';

export default function TestPage() {
  const { originalNepaliAlphabets, originalNslLetters } = useGlobalContext();
  
  // --- STATE ---
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  
  const [testState, setTestState] = useState<TestState>('idle');
  const [questionQueue, setQuestionQueue] = useState<number[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);

  const [predictionResult, setPredictionResult] = useState({ prediction: "--", confidence: 0 });
  const [feedback, setFeedback] = useState<'neutral' | 'correct'>('neutral');

  // --- REFS ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const requestRef = useRef<number | null>(null);
  const lastApiCallTimeRef = useRef<number>(0);
  const isPausedRef = useRef(false);
  
  // Ref to track state inside the Animation Loop
  const testStateRef = useRef<TestState>('idle');

  // Sync Ref with State
  useEffect(() => {
    testStateRef.current = testState;
  }, [testState]);

  // --- HELPER: Shuffle & Setup ---
  const startNewTest = () => {
    const indices = Array.from({ length: originalNepaliAlphabets.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const selectedQuestions = indices.slice(0, QUESTIONS_PER_TEST);

    setQuestionQueue(selectedQuestions);
    setCurrentIndex(0);
    setScore(0);
    setStartTime(Date.now());
    setFeedback('neutral');
    
    setTestState('running');
    testStateRef.current = 'running';
    
    if (!cameraActive) startCamera();
  };

  const handleStopTest = () => {
    setTestState('idle');
    setFeedback('neutral');
    stopCamera();
  };

  // --- API MUTATION ---
  const { mutate: predictSign } = useMutation({
    mutationFn: predictionApi.predict,
    onSuccess: (data) => {
      setPredictionResult(data);

      if (testStateRef.current !== 'running') return;

      const currentRealIndex = questionQueue[currentIndex];
      const targetChar = originalNslLetters[currentRealIndex];

      if (data.prediction === targetChar && data.confidence >= 0.95 && !isPausedRef.current) {
        isPausedRef.current = true;
        setFeedback('correct');
        setScore(prev => prev + 1);

        setTimeout(() => {
          if (currentIndex < QUESTIONS_PER_TEST - 1) {
            setCurrentIndex(prev => prev + 1);
            setFeedback('neutral');
            isPausedRef.current = false;
          } else {
            setEndTime(Date.now());
            setTestState('completed');
            stopCamera(); 
            isPausedRef.current = false;
          }
        }, 1500);
      }
    },
    onError: (err) => console.error("Prediction Error:", err)
  });

  const normalizeLandmarks = (landmarks: any[]) => {
    const landmarkArray = landmarks.map((lm) => [lm.x, lm.y, lm.z]);
    const center = landmarkArray.reduce((acc, [x, y, z]) => [acc[0] + x, acc[1] + y, acc[2] + z], [0, 0, 0]).map((sum) => sum / landmarkArray.length);
    const normalized = landmarkArray.map(([x, y, z]) => [x - center[0], y - center[1], z - center[2]]);
    const maxDistance = Math.max(...normalized.map(([x, y, z]) => Math.sqrt(x * x + y * y + z * z)));
    return normalized.map(([x, y, z]) => [x / maxDistance, y / maxDistance, z / maxDistance]);
  };

  // --- SETUP ---
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

  // --- LOOP ---
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
        if (testStateRef.current === 'running' && !isPausedRef.current && now - lastApiCallTimeRef.current >= 1000) {
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
    // 1. Cancel Loop
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
      requestRef.current = null;
    }
    
    // 2. Stop Hardware
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    // 3. FIX: Clear the Canvas (Remove red/green lines)
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    setCameraActive(false);
  };

  const getCurrentQuestionChar = () => {
    if (questionQueue.length === 0) return { nepali: '?', nsl: '?' };
    const realIndex = questionQueue[currentIndex];
    return {
      nepali: originalNepaliAlphabets[realIndex],
      nsl: originalNslLetters[realIndex]
    };
  };

  const calculateTimeTaken = () => {
    const diff = (endTime - startTime) / 1000;
    const minutes = Math.floor(diff / 60);
    const seconds = Math.floor(diff % 60);
    return `${minutes}m ${seconds}s`;
  };

  return (
    <div className="h-full w-full grid grid-cols-1 lg:grid-cols-2 gap-4 p-2 overflow-hidden]">
      
      {/* LEFT PANEL */}
      <div className="flex-1 flex flex-col bg-white dark:bg-neutral-900 rounded-3xl p-8 shadow-sm border border-border relative overflow-hidden">
        
        {testState === 'idle' && (
          <div className="flex-1 flex flex-col items-center justify-center text-center space-y-6">
            <div className="w-20 h-20 bg-primary/10 rounded-full flex items-center justify-center mb-2">
              <Trophy size={40} className="text-primary" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-foreground">Skill Test</h1>
              <p className="text-muted-foreground mt-2 max-w-xs mx-auto">
                You will be tested on {QUESTIONS_PER_TEST} random characters. Try to replicate them as fast as you can.
              </p>
            </div>
            <button onClick={startNewTest} disabled={!isModelLoaded} className="flex items-center gap-2 bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-3 rounded-full font-bold text-lg transition-all shadow-lg hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed">
              {!isModelLoaded ? <Loader2 className="animate-spin" /> : <Play size={20} fill="currentColor" />}
              <span>Start Test</span>
            </button>
            {!isModelLoaded && <span className="text-xs text-muted-foreground">Waiting for AI Model...</span>}
          </div>
        )}

        {testState === 'running' && (
          <div className="flex-1 flex flex-col w-full">
            <div className="flex justify-between items-center w-full mb-8">
              <div className="flex flex-col">
                <span className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Progress</span>
                <span className="text-2xl font-bold text-foreground">{currentIndex + 1} <span className="text-muted-foreground text-lg">/ {QUESTIONS_PER_TEST}</span></span>
              </div>
              <button onClick={handleStopTest} className="px-4 py-2 bg-red-50 dark:bg-red-900/20 text-red-500 rounded-lg text-sm font-medium hover:bg-red-100 dark:hover:bg-red-900/40 transition-colors">Quit Test</button>
            </div>

            <div className="flex-1 flex flex-col items-center justify-center relative">
              <div className="absolute top-0 px-4 py-1.5 bg-secondary rounded-full text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                Sign: {getCurrentQuestionChar().nsl}
              </div>
              <div className={cn("text-[5rem] md:text-[10rem] font-bold leading-none transition-all duration-300", feedback === 'correct' ? "text-green-500 scale-110" : "text-foreground")}>
                {getCurrentQuestionChar().nepali}
              </div>
              {feedback === 'correct' && (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                  <CheckCircle2 size={150} className="text-green-500/20 animate-ping" />
                </div>
              )}
            </div>

            <div className="mt-8 grid grid-cols-2 gap-4">
               <div className="bg-secondary/30 p-3 rounded-2xl flex flex-col items-center">
                 <span className="text-[10px] uppercase font-bold text-muted-foreground">Prediction</span>
                 <span className="text-xl font-bold text-blue-500">{predictionResult.prediction}</span>
               </div>
               <div className="bg-secondary/30 p-3 rounded-2xl flex flex-col items-center">
                 <span className="text-[10px] uppercase font-bold text-muted-foreground">Confidence</span>
                 <span className="text-xl font-bold text-blue-500">{(predictionResult.confidence * 100).toFixed(0)}%</span>
               </div>
            </div>
          </div>
        )}

        {testState === 'completed' && (
          <div className="flex-1 flex flex-col items-center justify-center text-center space-y-8">
            <div className="relative">
               <div className="absolute inset-0 bg-yellow-400 blur-2xl opacity-20 rounded-full"></div>
               <Trophy size={80} className="text-yellow-500 relative z-10" />
            </div>
            <div>
              <h2 className="text-4xl font-bold text-foreground">Test Completed!</h2>
              <p className="text-muted-foreground mt-2">Great job practicing your signs.</p>
            </div>
            <div className="grid grid-cols-2 gap-4 w-full max-w-xs">
              <div className="bg-secondary/50 p-4 rounded-2xl flex flex-col items-center">
                <span className="text-3xl font-bold text-foreground">{score}</span>
                <span className="text-xs uppercase font-bold text-muted-foreground mt-1">Questions</span>
              </div>
              <div className="bg-secondary/50 p-4 rounded-2xl flex flex-col items-center">
                <span className="text-3xl font-bold text-foreground">{calculateTimeTaken()}</span>
                <span className="text-xs uppercase font-bold text-muted-foreground mt-1">Time</span>
              </div>
            </div>
            <button onClick={startNewTest} className="flex items-center gap-2 bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-3 rounded-full font-bold text-lg transition-all shadow-lg hover:scale-105 active:scale-95">
              <RotateCcw size={20} />
              <span>Test Again</span>
            </button>
          </div>
        )}
      </div>

      {/* RIGHT PANEL: CAMERA */}
      <div className="h-full min-h-64 bg-black rounded-3xl overflow-hidden relative shadow-lg flex items-center justify-center group border border-neutral-800">
        {!isModelLoaded && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white bg-neutral-900 z-30">
            <Loader2 className="w-8 h-8 animate-spin text-blue-500 mb-2" />
            <p className="text-xs font-medium">Loading Vision...</p>
          </div>
        )}
        
        {/* FIX: Show prompt when idle OR completed (since camera stops in both) */}
        {isModelLoaded && !cameraActive && (testState === 'idle' || testState === 'completed') && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white bg-neutral-900/80 backdrop-blur-sm z-30">
             <div className="w-20 h-20 bg-neutral-800 rounded-full flex items-center justify-center mb-4">
               <Camera size={32} className="text-neutral-400" />
             </div>
             <p className="text-neutral-400 font-medium">Camera starts with test</p>
          </div>
        )}

        <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover opacity-80" playsInline muted autoPlay style={{ transform: "scaleX(-1)" }} />
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover z-20" />
        
        {cameraActive && (
           <div className="absolute top-4 right-4 z-30 bg-black/60 px-3 py-1.5 rounded-full flex items-center gap-2 border border-white/10 shadow-xl pointer-events-none">
             <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
             <span className="text-white text-xs font-bold uppercase tracking-wider">Exam Live</span>
           </div>
        )}
      </div>

    </div>
  );
}