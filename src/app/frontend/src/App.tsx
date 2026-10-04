import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Shell } from './components/layout/Shell';
import { UniversalRestoration } from './pages/UniversalRestoration';
import { HardRoutedRestoration } from './pages/HardRoutedRestoration';
import { SoftMoERestoration } from './pages/SoftMoERestoration';
import { FaceToSketch } from './pages/FaceToSketch';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Shell>
        <Routes>
          <Route path="/" element={<Navigate to="/universal" replace />} />
          <Route path="/universal" element={<UniversalRestoration />} />
          <Route path="/hard-routed" element={<HardRoutedRestoration />} />
          <Route path="/soft-moe" element={<SoftMoERestoration />} />
          <Route path="/face-to-sketch" element={<FaceToSketch />} />
          <Route path="*" element={<Navigate to="/universal" replace />} />
        </Routes>
      </Shell>
    </BrowserRouter>
  );
};

export default App;
