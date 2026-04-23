import { useEffect, useRef, useState } from 'react';

interface SplashScreenProps {
  onComplete: () => void;
}

export function SplashScreen({ onComplete }: SplashScreenProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [phase, setPhase] = useState<'video' | 'fadeout' | 'done'>('video');

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    // When video ends, start fade to white
    const handleEnded = () => {
      setPhase('fadeout');
      setTimeout(() => {
        setPhase('done');
        onComplete();
      }, 1000); // 1s fade duration
    };

    // Fallback: if video fails to load, skip after 5s
    const handleError = () => {
      setPhase('fadeout');
      setTimeout(() => {
        setPhase('done');
        onComplete();
      }, 1000);
    };

    video.addEventListener('ended', handleEnded);
    video.addEventListener('error', handleError);

    // Auto-play
    video.play().catch(() => {
      // Autoplay blocked — skip intro
      handleError();
    });

    return () => {
      video.removeEventListener('ended', handleEnded);
      video.removeEventListener('error', handleError);
    };
  }, [onComplete]);

  if (phase === 'done') return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        backgroundColor: '#000',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
      }}
    >
      {/* Video */}
      <video
        ref={videoRef}
        src="/GateKeeperFin.mp4"
        muted
        playsInline
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          opacity: phase === 'fadeout' ? 0 : 1,
          transition: 'opacity 1s ease-in-out',
        }}
      />

      {/* White fade overlay */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundColor: '#ffffff',
          opacity: phase === 'fadeout' ? 1 : 0,
          transition: 'opacity 1s ease-in-out',
          pointerEvents: 'none',
        }}
      />
    </div>
  );
}