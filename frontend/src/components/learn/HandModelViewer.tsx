'use client';

import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF, Stage, Html } from '@react-three/drei';
import { Loader2 } from 'lucide-react';

// --- THE MODEL COMPONENT ---
function Model({ url }: { url: string }) {
  // useGLTF will auto-cache the model
  const { scene } = useGLTF(url);
  return <primitive object={scene} />;
}

// --- FALLBACK PLACEHOLDER (When no model exists) ---
function PlaceholderModel() {
  return (
    <mesh rotation={[0, 0, 0]}>
      <boxGeometry args={[2, 3, 0.5]} />
      <meshStandardMaterial color="#cbd5e1" wireframe />
      <Html position={[0, 0, 0]} center>
        <div className="bg-black/80 text-white px-2 py-1 rounded text-xs whitespace-nowrap">
          No 3D Model Available
        </div>
      </Html>
    </mesh>
  );
}

// --- MAIN VIEWER ---
interface HandModelViewerProps {
  character: string; // e.g., 'Ka', 'Chha'
}

// Map characters to your actual .glb files in /public/models/
const MODEL_MAP: Record<string, string> = {
  'Ka': '/models/ka.glb',
  'Cha': '/models/cha.glb',
  'Chha': '/models/chhagaurav.glb', // Your existing model
  'Ga': '/models/ga.glb', 
  // Add others later: 'Ka': '/models/ka.glb',
};

export default function HandModelViewer({ character }: HandModelViewerProps) {
  const modelUrl = MODEL_MAP[character];

  return (
    <div className="w-full h-full bg-neutral-100 dark:bg-neutral-800/50 rounded-3xl overflow-hidden relative border border-border">
      
      <div className="absolute bottom-2 left-2 z-10 px-3 py-1 bg-white/90 dark:bg-black/50 backdrop-blur rounded-full text-xs font-semibold uppercase tracking-wider text-muted-foreground shadow-sm">
        3D View • Rotate to Explore
      </div>

      <Canvas shadows dpr={[1, 2]} camera={{ fov: 50, position: [0, 0, 5] }}>
        <Suspense fallback={<Loader />}>
          <Stage environment="city" intensity={0.6}>
            {modelUrl ? (
              <Model url={modelUrl} />
            ) : (
              <PlaceholderModel />
            )}
          </Stage>
        </Suspense>
        <OrbitControls  makeDefault />
      </Canvas>
    </div>
  );
}

// Simple internal loader for the Canvas
function Loader() {
  return (
    <Html center>
      <div className="flex flex-col items-center gap-2 text-muted-foreground">
        <Loader2 className="animate-spin w-6 h-6" />
        <span className="text-xs font-medium">Loading 3D...</span>
      </div>
    </Html>
  );
}