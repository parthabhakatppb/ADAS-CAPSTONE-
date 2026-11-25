import React, { useState } from 'react';
import { useAdasStream } from './hooks/useAdasStream';
import TopBar from './components/TopBar';
import ViewportHUD from './components/ViewportHUD';
import TelemetrySidebar from './components/TelemetrySidebar';
import BottomDock from './components/BottomDock';

const App: React.FC = () => {
  const stream = useAdasStream();
  const [muted, setMuted] = useState(true);

  return (
    <div className="flex flex-col h-screen bg-primary text-white overflow-hidden">
      <TopBar
        mode={stream.mode}
        onModeChange={stream.setMode}
        connected={stream.connected}
        modelVariant={stream.modelVariant}
        onModelChange={stream.setModelVariant}
        precision={stream.precision}
        onPrecisionChange={stream.setPrecision}
        weatherOverride={stream.weatherOverride}
        onWeatherChange={stream.setWeatherOverride}
        scenario={stream.scenario}
        onScenarioChange={stream.setScenario}
        muted={muted}
        onMuteToggle={() => setMuted(!muted)}
      />
      <div className="flex flex-1 overflow-hidden min-h-0">
        <ViewportHUD
          canvasRef={stream.canvasRef}
          videoRef={stream.videoRef}
          payload={stream.payload}
          mode={stream.mode}
          cameraError={stream.cameraError}
          videoFileName={stream.videoFileName}
          onVideoFileSelect={stream.loadVideoFile}
        />
        <TelemetrySidebar payload={stream.payload} />
      </div>
      <BottomDock
        logs={stream.logs}
        onClearLogs={() => stream.setLogs([])}
        payload={stream.payload}
        mode={stream.mode}
        videoFileName={stream.videoFileName}
        isPlaying={stream.isPlaying}
        videoProgress={stream.videoProgress}
        videoDuration={stream.videoDuration}
        playbackRate={stream.playbackRate}
        onTogglePlay={stream.togglePlay}
        onSkipBack={() => stream.skipBy(-5)}
        onSkipForward={() => stream.skipBy(5)}
        onSeek={stream.seekTo}
        onPlaybackRateChange={stream.changePlaybackRate}
      />
    </div>
  );
};

export default App;
