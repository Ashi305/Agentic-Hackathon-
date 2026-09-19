from dotenv import load_dotenv
load_dotenv()

import os


from agent_orchestrator import SafetyReportAgent
from few_shot_manager import FewShotManager
from mock_data import SAMPLE_REPORTS, MOCK_CORRECTIONS


def setup_few_shot_manager():

    manager = FewShotManager()

    for correction in MOCK_CORRECTIONS:

        manager.add_correction(
            report=correction["report"],
            original_label=correction["original_label"],
            corrected_label=correction["corrected_label"],
            reason=correction["reason"]
        )

    return manager


def print_result(result):

    print("\n" + "=" * 60)
    print("SAFETY REPORT")
    print("=" * 60)

    print(result["report"])

    print("\nEXTRACTION")
    print("-" * 60)

    extraction = result["extraction"]

    for key, value in extraction.items():
        print(f"{key}: {value}")

    print("\nRISK CLASSIFICATION")
    print("-" * 60)

    classification = result["classification"]

    print(
        f"Risk level: "
        f"{classification['risk_level']}"
    )

    print(
        f"Confidence: "
        f"{classification['confidence']}"
    )

    print(
        f"Reason: "
        f"{classification['reason']}"
    )

    print(
        f"Human review required: "
        f"{classification['human_review_required']}"
    )


def main():

    load_dotenv()

    few_shot_manager = setup_few_shot_manager()

    agent = SafetyReportAgent(
        few_shot_manager
    )

    report = SAMPLE_REPORTS[1]

    result = agent.analyze(report)

    print_result(result)


if __name__ == "__main__":
    main()