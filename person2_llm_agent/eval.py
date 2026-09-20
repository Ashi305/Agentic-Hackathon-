import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix

load_dotenv()

from agent_orchestrator import SafetyReportAgent
from few_shot_manager import FewShotManager

CSV_FILE = "report.csv"
N_SAMPLES = 50
LABEL_MAP = {"ordinary": "LOW", "critical": "HIGH"}


def setup_agent():
    return SafetyReportAgent(FewShotManager())


def main():
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=["Report_Text", "Ground_Truth_Severity"])
    df = df.sample(n=min(N_SAMPLES, len(df)), random_state=42)  # reproducible

    print(f"Loaded {len(df)} reports.")
    agent = setup_agent()

    y_true, y_pred = [], []
    errors, skipped = 0, 0

    for n, (_, row) in enumerate(df.iterrows(), start=1):
        actual = LABEL_MAP.get(str(row["Ground_Truth_Severity"]).strip().lower())
        if actual is None:
            skipped += 1
            print(f"{n}: skipped unknown label {row['Ground_Truth_Severity']!r}")
            continue

        try:
            result = agent.analyze(row["Report_Text"])
            predicted = str(result["classification"]["risk_level"]).strip().upper()
        except Exception as e:
            errors += 1
            print(f"{n}: ERROR: {e}")
            continue

        y_true.append(actual)
        y_pred.append(predicted)
        print(f"{n}/{len(df)} | Actual: {actual} | Predicted: {predicted}")

    total = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / total if total else 0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Evaluated: {total} | Errors: {errors} | Skipped: {skipped}")
    print(f"Correct: {correct} | Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    if total:
        labels = sorted(set(y_true) | set(y_pred))
        print("\nActual distribution:\n", pd.Series(y_true).value_counts())
        print("\nPredicted distribution:\n", pd.Series(y_pred).value_counts())
        print("\nConfusion matrix (rows=actual, cols=predicted):")
        print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                           index=labels, columns=labels))
        print("\n", classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()