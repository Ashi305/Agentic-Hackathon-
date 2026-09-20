import json
import time
import pandas as pd
from dotenv import load_dotenv

from agent_orchestrator import SafetyReportAgent
from few_shot_manager import FewShotManager


# -----------------------------
# SETTINGS
# -----------------------------

CSV_FILE = "report.csv"

START_ROW = 0
NUM_REPORTS = 5

DELAY_BETWEEN_REPORTS = 10

OUTPUT_FILE = "results.json"


# -----------------------------
# SET UP AGENT
# -----------------------------

def setup_agent():
    manager = FewShotManager()
    return SafetyReportAgent(manager)


# -----------------------------
# MAIN
# -----------------------------

def main():

    load_dotenv()

    # Load CSV
    df = pd.read_csv(CSV_FILE)

    # Remove reports with no text
    df = df.dropna(subset=["Report_Text"])

    end_row = min(
        START_ROW + NUM_REPORTS,
        len(df)
    )

    reports = df.iloc[START_ROW:end_row]

    print(f"Total reports in CSV: {len(df)}")
    print(
        f"Processing rows "
        f"{START_ROW} to {end_row - 1}"
    )
    print(f"Reports this run: {len(reports)}")
    print()

    agent = setup_agent()

    results = []

    # -----------------------------
    # PROCESS REPORTS
    # -----------------------------

    for count, (_, row) in enumerate(
        reports.iterrows(),
        start=1
    ):

        report_number = START_ROW + count

        print("=" * 60)
        print(f"REPORT {report_number}/{len(df)}")
        print("=" * 60)

        try:

            result = agent.analyze(
                row["Report_Text"]
            )

            results.append(result)

            classification = result["classification"]

            print(
                f"Risk: "
                f"{classification['risk_level']}"
            )

            print(
                f"Confidence: "
                f"{classification['confidence']}"
            )

        except Exception as e:

            print("\nERROR:")
            print(e)

            error_text = str(e)

            # Stop immediately if quota is exhausted
            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):
                print("\nGemini quota reached.")
                print("Stopping to avoid more requests.")
                break

            # Network connection problem
            if (
                "10053" in error_text
                or "connection" in error_text.lower()
            ):
                print(
                    "\nNetwork error."
                    " Skipping this report."
                )
                continue

            print("Skipping this report.")

        # Wait before the next report
        if count < len(reports):

            print(
                f"\nWaiting "
                f"{DELAY_BETWEEN_REPORTS} seconds..."
            )

            time.sleep(DELAY_BETWEEN_REPORTS)

    # -----------------------------
    # SAVE RESULTS
    # -----------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Successfully processed: "
        f"{len(results)}"
    )

    print(
        f"Results saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()