import { BrainCircuit, Send } from "lucide-react";

export default function QuestionPanel({ answers, disabled, needsQuestions, onAnswer, onSubmit, questions }) {
  return (
    <section className={`panel question-panel ${needsQuestions ? "attention" : ""}`} aria-labelledby="questions-title">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Signal two</p>
          <h2 id="questions-title">Feeling check</h2>
        </div>
        <BrainCircuit size={24} />
      </div>

      <div className="question-list">
        {questions.map((question) => {
          const value = answers[question.id] ?? 3;
          const sliderProgress = `${((value - 1) / 4) * 100}%`;
          return (
            <label className="slider-field" key={question.id}>
              <span>{question.prompt}</span>
              <input
                type="range"
                min="1"
                max="5"
                step="1"
                value={value}
                style={{ "--slider-progress": sliderProgress }}
                onChange={(event) => onAnswer(question.id, Number(event.target.value))}
                disabled={disabled}
              />
              <span className="scale-labels">
                <small>{question.lowLabel}</small>
                <strong>{value}</strong>
                <small>{question.highLabel}</small>
              </span>
            </label>
          );
        })}
      </div>

      <button className="icon-button primary wide" onClick={onSubmit} disabled={disabled}>
        <Send size={18} />
        Analyze
      </button>
    </section>
  );
}
