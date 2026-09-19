SAMPLE_REPORTS = [

    "A worker was walking through the warehouse and noticed "
    "oil spilled across the main walkway. No warning sign was placed.",

    "An employee noticed exposed electrical wiring near a "
    "water leak in the maintenance area.",

    "A worker forgot to wear safety glasses while handling "
    "a small piece of metal.",

    "Several boxes were stacked too close to the edge of a shelf "
    "and one box nearly fell onto an employee.",

    "The emergency exit was partially blocked by stored equipment.",

    "A forklift was operating near pedestrians in a busy warehouse "
    "with no clear separation between the walking path and vehicle path."
]


MOCK_CORRECTIONS = [

    {
        "report": (
            "Exposed electrical wiring was found next to "
            "standing water in the maintenance area."
        ),
        "original_label": "MEDIUM",
        "corrected_label": "HIGH",
        "reason": (
            "Electrical exposure combined with water creates "
            "a serious electric shock hazard."
        )
    },

    {
        "report": (
            "A forklift was operating close to pedestrians "
            "without a separated walking path."
        ),
        "original_label": "MEDIUM",
        "corrected_label": "HIGH",
        "reason": (
            "Interaction between moving vehicles and pedestrians "
            "creates potential for serious injury."
        )
    },

    {
        "report": (
            "A small amount of oil was found on a walkway "
            "and a warning sign was placed immediately."
        ),
        "original_label": "HIGH",
        "corrected_label": "MEDIUM",
        "reason": (
            "The spill creates a slip hazard, but immediate "
            "controls reduced the level of risk."
        )
    }
]