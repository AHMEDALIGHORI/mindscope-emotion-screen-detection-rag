import { useEffect, useRef, useState } from "react";
import { Camera, CircleStop, ScanFace, Video } from "lucide-react";

const FRAME_COUNT = 6;
const FRAME_DELAY_MS = 140;

export default function CameraPanel({ capturedImage, disabled, onCapture }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [isLive, setIsLive] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [cameraError, setCameraError] = useState("");

  useEffect(() => {
    return () => stopCamera();
  }, []);

  async function startCamera() {
    setCameraError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setIsLive(true);
    } catch (error) {
      setCameraError(error.message || "Camera permission failed.");
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsLive(false);
    setIsScanning(false);
    setScanProgress(0);
  }

  async function captureBurst() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) {
      setCameraError("Camera frame is not ready yet.");
      return;
    }

    setIsScanning(true);
    setCameraError("");
    const frames = [];
    for (let index = 0; index < FRAME_COUNT; index += 1) {
      frames.push(captureFrame());
      setScanProgress(index + 1);
      if (index < FRAME_COUNT - 1) {
        await sleep(FRAME_DELAY_MS);
      }
    }
    setIsScanning(false);
    await onCapture(frames);
  }

  function captureFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.88);
  }

  const status = isScanning ? `${scanProgress}/${FRAME_COUNT}` : isLive ? "Live" : "Idle";

  return (
    <section className="panel camera-panel" aria-labelledby="camera-title">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Signal one</p>
          <h2 id="camera-title">Face scan</h2>
        </div>
        <span className={isLive ? "status live" : "status"}>{status}</span>
      </div>

      <div className={isScanning ? "camera-frame scanning" : "camera-frame"}>
        <video ref={videoRef} playsInline muted aria-label="Camera preview" />
        {!isLive && capturedImage ? <img src={capturedImage} alt="Captured face frame" /> : null}
        {!isLive && !capturedImage ? (
          <div className="camera-placeholder">
            <ScanFace size={44} />
            <span>Ready</span>
          </div>
        ) : null}
        <div className="scan-reticle" aria-hidden="true">
          <span className="corner top-left" />
          <span className="corner top-right" />
          <span className="corner bottom-left" />
          <span className="corner bottom-right" />
          <span className="radar-ring ring-a" />
          <span className="radar-ring ring-b" />
          <span className="focus-dot" />
        </div>
      </div>

      <div className="button-row">
        <button className="icon-button primary" onClick={startCamera} disabled={disabled || isLive}>
          <Video size={18} />
          Start
        </button>
        <button className="icon-button" onClick={captureBurst} disabled={disabled || !isLive || isScanning}>
          <Camera size={18} />
          Scan
        </button>
        <button className="icon-button ghost" onClick={stopCamera} disabled={!isLive}>
          <CircleStop size={18} />
          Stop
        </button>
      </div>

      <div className="scan-note">
        <span>Multi-frame scan</span>
        <strong>{FRAME_COUNT} frames</strong>
      </div>

      {cameraError ? <p className="inline-error">{cameraError}</p> : null}
      <canvas ref={canvasRef} hidden />
    </section>
  );
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
