import { useEffect, useMemo, useState } from "react";
import { BookOpen, Brain, Database, Layers3, RefreshCw, ScanLine, ShieldCheck } from "lucide-react";
import CameraPanel from "./components/CameraPanel.jsx";
import QuestionPanel from "./components/QuestionPanel.jsx";
import ResultPanel from "./components/ResultPanel.jsx";
import { analyzeEmotion, getHealth, getQuestions, retrainModel } from "./lib/api.js";

export default function App() {
  const [answers, setAnswers] = useState({});
  const [capturedImage, setCapturedImage] = useState("");
  const [health, setHealth] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let ignore = false;
    async function load() {
      try {
        const [healthResponse, questionResponse] = await Promise.all([getHealth(), getQuestions()]);
        if (!ignore) {
          setHealth(healthResponse);
          setQuestions(questionResponse.questions);
        }
      } catch (requestError) {
        if (!ignore) setError(apiMessage(requestError));
      }
    }
    load();
    return () => {
      ignore = true;
    };
  }, []);

  const answeredCount = useMemo(() => Object.keys(answers).length, [answers]);

  function updateAnswer(questionId, value) {
    setAnswers((current) => ({ ...current, [questionId]: value }));
  }

  async function runAnalysis(nextImages = null) {
    setIsLoading(true);
    setError("");
    try {
      const response = await analyzeEmotion({
        image: nextImages ? undefined : capturedImage,
        images: nextImages ?? undefined,
        answers,
      });
      setResult(response);
    } catch (requestError) {
      setError(apiMessage(requestError));
    } finally {
      setIsLoading(false);
    }
  }

  async function handleCapture(images) {
    const lastImage = images[images.length - 1];
    setCapturedImage(lastImage);
    await runAnalysis(images);
  }

  async function handleRetrain() {
    setIsLoading(true);
    setError("");
    try {
      await retrainModel();
      const response = await getHealth();
      setHealth(response);
    } catch (requestError) {
      setError(apiMessage(requestError));
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark">
            <Brain size={25} />
          </span>
          <div>
            <h1>MindScope</h1>
            <p>Emotion screen detection</p>
          </div>
        </div>
        <div className="topbar-actions">
          <span className="health-pill">{health ? `${Math.round(health.metrics.ensemble_accuracy * 100)}% model` : "API"}</span>
          <span className="health-pill">{health ? `${health.metrics.engineered_feature_count ?? 21} features` : "RAG"}</span>
          <button className="icon-button ghost" onClick={handleRetrain} disabled={isLoading}>
            <RefreshCw size={17} />
            Train
          </button>
        </div>
      </header>

      <section className="insight-ribbon" aria-label="System overview">
        <div className="insight-card active">
          <Database size={18} />
          <span>Dataset fusion</span>
          <strong>FER + mood</strong>
        </div>
        <div className="insight-card">
          <Layers3 size={18} />
          <span>Model stack</span>
          <strong>Forest + 3x9 NN</strong>
        </div>
        <div className="insight-card">
          <BookOpen size={18} />
          <span>Context RAG</span>
          <strong>Emotion books</strong>
        </div>
        <div className="insight-card">
          <ShieldCheck size={18} />
          <span>Readiness</span>
          <strong>{health ? "API online" : "Checking"}</strong>
        </div>
      </section>

      {error ? (
        <div className="app-error" role="alert">
          {error}
        </div>
      ) : null}

      <section className="workspace-grid">
        <div className="input-stack">
          <CameraPanel capturedImage={capturedImage} disabled={isLoading} onCapture={handleCapture} />
          <QuestionPanel
            answers={answers}
            disabled={isLoading || questions.length === 0}
            needsQuestions={Boolean(result?.needsQuestions)}
            onAnswer={updateAnswer}
            onSubmit={() => runAnalysis()}
            questions={questions}
          />
        </div>

        <ResultPanel result={result} />
      </section>

      <footer className="footer-strip">
        <span>
          <ScanLine size={16} />
          {answeredCount} feeling signals selected
        </span>
        <span>Kaggle-ready FER import + emotion-book RAG + weighted model stack</span>
      </footer>

      {isLoading ? <div className="loading-scrim">Analyzing</div> : null}
    </main>
  );
}

function apiMessage(error) {
  const message = error?.message || "Something went wrong.";
  if (message.includes("Failed to fetch")) {
    return "API is not running on http://127.0.0.1:8000.";
  }
  return message;
}
