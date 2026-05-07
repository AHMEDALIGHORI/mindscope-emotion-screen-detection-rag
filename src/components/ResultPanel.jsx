import { Activity, AlertCircle, BookOpen, Film, Gauge, Laugh, Layers3, Network, ShieldCheck } from "lucide-react";

const icons = {
  movie: Film,
  book: BookOpen,
  joke: Laugh,
};

export default function ResultPanel({ result }) {
  if (!result) {
    return (
      <section className="panel result-panel empty" aria-label="Emotion result">
        <div className="empty-state">
          <div className="empty-orbit">
            <Activity size={34} />
            <span />
            <span />
          </div>
          <div className="empty-copy">
            <p className="eyebrow">Analysis chamber</p>
            <h2>Awaiting signal</h2>
          </div>
          <div className="empty-signal-grid">
            <span>
              <Gauge size={16} />
              <strong>Camera</strong>
              <small>Standby</small>
            </span>
            <span>
              <Network size={16} />
              <strong>RAG</strong>
              <small>Indexed</small>
            </span>
            <span>
              <BookOpen size={16} />
              <strong>Books</strong>
              <small>Ready</small>
            </span>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="panel result-panel" aria-label="Emotion result">
      <div className="result-hero">
        <div>
          <p className="eyebrow">Detected state</p>
          <h2>{titleCase(result.emotion)}</h2>
          <p className="confidence">{Math.round(result.confidence * 100)}% calibrated confidence</p>
        </div>
        <div className="emotion-orbit" aria-label={`${result.emotion} confidence`}>
          <span>{Math.round(result.confidence * 100)}</span>
        </div>
      </div>

      <div className="result-snapshot">
        <div>
          <span>Quality</span>
          <strong>{titleCase(result.analysisQuality.label)}</strong>
        </div>
        <div>
          <span>Frames</span>
          <strong>{`${result.face.detectedFrames ?? 0}/${result.face.frameCount ?? 0}`}</strong>
        </div>
        <div>
          <span>Cluster</span>
          <strong>{`#${result.forestCluster.id}`}</strong>
        </div>
      </div>

      {result.needsQuestions ? (
        <div className="notice">
          <AlertCircle size={18} />
          <span>{result.questionReason}</span>
        </div>
      ) : null}

      <div className="probabilities">
        {result.probabilities.slice(0, 5).map((item) => (
          <div className="probability-row" key={item.emotion}>
            <span>{titleCase(item.emotion)}</span>
            <div className="meter">
              <i style={{ width: `${Math.max(4, item.score * 100)}%` }} />
            </div>
            <strong>{Math.round(item.score * 100)}%</strong>
          </div>
        ))}
      </div>

      <div className="signal-strip">
        {result.signals.map((signal) => (
          <span key={signal}>{signal}</span>
        ))}
      </div>

      <div className="model-grid">
        <Metric label="Ensemble" value={percent(result.model.metrics.ensemble_accuracy)} />
        <Metric label="Forest" value={percent(result.model.metrics.random_forest_accuracy)} />
        <Metric label="ExtraTrees" value={percent(result.model.metrics.extra_trees_accuracy)} />
        <Metric label="Boost" value={percent(result.model.metrics.gradient_boost_accuracy)} />
        <Metric label="3x9 NN" value={percent(result.model.metrics.neural_network_accuracy)} />
        <Metric label="Quality" value={titleCase(result.analysisQuality.label)} />
        <Metric label="Margin" value={percent(result.uncertainty.margin)} />
        <Metric label="Cluster" value={`#${result.forestCluster.id}`} />
      </div>

      <div className="diagnostic-band">
        <div className="mini-heading">
          <Gauge size={18} />
          <span>Signal quality</span>
        </div>
        <div className="diagnostic-grid">
          <Diagnostic label="Face" value={percent(result.analysisQuality.faceQuality)} />
          <Diagnostic label="Answers" value={percent(result.analysisQuality.answerQuality)} />
          <Diagnostic label="Agreement" value={percent(result.uncertainty.agreement)} />
          <Diagnostic label="Frames" value={`${result.face.detectedFrames ?? 0}/${result.face.frameCount ?? 0}`} />
        </div>
      </div>

      <div className="weight-band">
        <div className="mini-heading">
          <Layers3 size={18} />
          <span>Model stack</span>
        </div>
        <div className="weight-list">
          {Object.entries(result.model.modelWeights ?? {}).map(([name, weight]) => (
            <span key={name}>
              {modelName(name)} <strong>{Math.round(Number(weight) * 100)}%</strong>
            </span>
          ))}
        </div>
      </div>

      <div className="rag-box">
        <div className="mini-heading">
          <Network size={18} />
          <span>Emotion books + vector context</span>
        </div>
        {result.ragContext.slice(0, 3).map((item) => (
          <p key={item.title}>{item.payload.text}</p>
        ))}
      </div>

      <div className="recommendation-grid">
        {Object.entries(result.recommendations).map(([type, items]) => {
          const Icon = icons[type];
          return (
            <div className="recommendation-column" key={type}>
              <div className="mini-heading">
                <Icon size={18} />
                <span>{titleCase(type)}s</span>
              </div>
              {items.map((item) => (
                <article className="recommendation-item" key={`${type}-${item.title}`}>
                  <h3>{item.title}</h3>
                  <p>{item.description}</p>
                  <small>{item.why}</small>
                </article>
              ))}
            </div>
          );
        })}
      </div>

      <div className="safety-note">
        <ShieldCheck size={17} />
        <span>{result.safetyNote}</span>
      </div>
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Diagnostic({ label, value }) {
  return (
    <div className="diagnostic">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function percent(value) {
  if (value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${Math.round(Number(value) * 100)}%`;
}

function titleCase(value) {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function modelName(value) {
  return value
    .split("_")
    .map((part) => (part === "net" ? "NN" : titleCase(part)))
    .join(" ");
}
