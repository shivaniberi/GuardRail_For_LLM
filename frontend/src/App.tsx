import { useState } from 'react';
import { ChatContainer } from './components/ChatContainer';
import { SplashScreen } from './components/SplashScreen';

function App() {
  const [showSplash, setShowSplash] = useState(true);

  return (
    <>
      {showSplash && (
        <SplashScreen onComplete={() => setShowSplash(false)} />
      )}
      <div
        style={{
          opacity: showSplash ? 0 : 1,
          transition: 'opacity 0.5s ease-in-out',
        }}
      >
        <ChatContainer />
      </div>
    </>
  );
}

export default App;