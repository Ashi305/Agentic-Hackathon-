"""
Person 1: Synthetic Data Generator
Generates a benchmark dataset of 40-50 realistic near-miss reports with granular risk factors,
locations, departments, and nuanced incident precursors across diverse industrial sectors.
"""
import json
import random
from datetime import datetime, timedelta
from typing import List
from .schema import (
    NearMissReport,
    StructuredExtraction,
    RiskAssessment,
    RiskLevel,
    HazardCategory,
    EnrichedReport,
)

DEPARTMENTS = [
    "Chemical Processing & Synthesis",
    "High-Bay Warehousing & Logistics",
    "Heavy Fabrication & Stamping",
    "Semiconductor Cleanroom Assembly",
    "Plant Facilities & Maintenance",
    "Fleet Loading Dock & Yard Operations",
]

FACILITIES = [
    "Plant Alpha - Midwest Complex",
    "Plant Beta - Gulf Coast Refinery",
    "Plant Gamma - Great Lakes Assembly",
    "Facility Delta - Apex Logistics Hub",
]

RAW_REPORTS_DATA = [
    {
        "dept": "Chemical Processing & Synthesis",
        "loc": "Reactor Bay 3 - Acid Dosing Skid",
        "role": "Chemical Operator II",
        "equip": "Sulfuric Acid Metering Pump P-104",
        "env": "High humidity, poor artificial lighting overhead",
        "action": "Isolated manual block valve V-12 and notified shift supervisor",
        "text": "During morning line flushing on Reactor 3, observed intermittent acid spitting from the PTFE flange gasket on metering pump P-104. Flange spray shield was improperly seated and secured with only a single zip-tie instead of required locking collar. Operator was wearing safety glasses instead of a full face shield. Acid droplets splashed onto sleeve of chemical apron but did not penetrate skin. If pressure had surged to normal 45 PSI dosing pressure, direct facial contact with 98% H2SO4 would have occurred.",
        "hazard": "Sulfuric Acid Line Leak with Displaced Flange Shield",
        "cat": HazardCategory.CHEMICAL_TOXIC,
        "precursors": ["Improperly secured spray shield", "Worn PTFE flange seal", "Inadequate eye/face PPE for line break"],
        "assets": ["Chemical Operator II", "P-104 Acid Skid"],
        "failed_safeguards": ["Missing full face shield", "Zip-tie used instead of approved locking flange band"],
        "mitigation": "Install bolted stainless steel flange shields; re-train on PPE SOP-CHEM-04.",
        "risk": RiskLevel.HIGH,
        "score": 8.9,
        "rationale": "High-concentration acid under pressure with missing primary physical barrier (spray shield) and PPE non-compliance. Catastrophic chemical burn potential.",
        "citations": ["OSHA 1910.1200 (Hazard Communication)", "OSHA 1910.133 (Eye and Face Protection)"],
        "signals": ["Chemical Precursor", "Inadequate PPE", "Pressurized Liquid Leak"],
        "escalate": True
    },
    {
        "dept": "High-Bay Warehousing & Logistics",
        "loc": "Aisle 14 - Cross-dock Intersection",
        "role": "Order Picker",
        "equip": "Toyota 3-Wheel Electric Forklift #7",
        "env": "Dry floor, high ambient noise from automated sorter",
        "action": "Pedestrian stepped back against racking; forklift driver applied emergency brake",
        "text": "Order picker walking out of Aisle 14 with hand truck was almost struck by Forklift #7 rounding the blind corner at speed. The ceiling-mounted parabolic convex mirror was twisted upwards and covered in dust, obscuring sightlines. The forklift driver's blue safety halo light was functioning, but the automatic horn interlock failed to sound upon entering the pedestrian intersection. Braking distance missed the pedestrian by less than 18 inches.",
        "hazard": "Blind Intersection Forklift-Pedestrian Near Collision",
        "cat": HazardCategory.MECHANICAL_CRUSH,
        "precursors": ["Misaligned/dirty convex intersection mirror", "Audible horn interlock malfunction", "Excessive speed approaching pedestrian crossing"],
        "assets": ["Order Picker", "Forklift Operator"],
        "failed_safeguards": ["Defective crossing horn", "Misdirected optical mirror"],
        "mitigation": "Reposition and lock parabolic mirrors; install automated radar-activated crossing gates and speed limiters.",
        "risk": RiskLevel.HIGH,
        "score": 9.1,
        "rationale": "High-speed mobile equipment near-miss with pedestrian in blind corridor; missing dual auditory/visual safeguards. Immediate severe crush/fatality precursor.",
        "citations": ["OSHA 1910.178 (Powered Industrial Trucks)"],
        "signals": ["Mobile Equipment Precursor", "Blind Corner", "Sub-2-Foot Near Miss"],
        "escalate": True
    },
    {
        "dept": "Heavy Fabrication & Stamping",
        "loc": "Press Line 2 - 400T Hydraulic Stamping Press",
        "role": "Die Setter",
        "equip": "Komatsu 400T Press PR-02",
        "env": "Oily mist, concrete floor with slip-resistant grit",
        "action": "Stopped machine cycling using e-stop and flagged maintenance",
        "text": "During progressive die setup, operator noticed optical light curtain (presence sensing device) intermittently failed to trip ram stoppage when a calibration test bar passed through beam 4 and 5. Alignment bracket on receiver pole was loose from vibration. Ram continued cycle while test wand was inside die space. Operator had both hands poised to guide sheet steel into nest before realizing curtain status lamp remained green.",
        "hazard": "Intermittent Light Curtain Failure on 400-Ton Hydraulic Press",
        "cat": HazardCategory.MECHANICAL_CRUSH,
        "precursors": ["Vibration-induced misalignment of safety sensor", "Missing daily light curtain wand verification sign-off"],
        "assets": ["Die Setter", "Tooling Assembly"],
        "failed_safeguards": ["Optical presence-sensing device (light curtain)", "Daily pre-shift bump test checklist"],
        "mitigation": "Torque and wire-lock mounting brackets; enforce strict pre-shift wand challenge verification interlock.",
        "risk": RiskLevel.HIGH,
        "score": 9.6,
        "rationale": "Direct mechanical defeat of point-of-operation safeguard on a heavy stamping press. Classic precursor to catastrophic amputation.",
        "citations": ["OSHA 1910.212 (General Requirements for all Machines)", "OSHA 1910.217 (Mechanical Power Presses)"],
        "signals": ["Amputation Precursor", "Safety Circuit Defeat", "Critical Point-of-Operation Failure"],
        "escalate": True
    },
    {
        "dept": "Plant Facilities & Maintenance",
        "loc": "Boiler Room Mezzanine Level 2",
        "role": "HVAC Mechanic",
        "equip": "Chilled Water Expansion Tank TK-08",
        "env": "Elevated temperature (95°F), cramped overhead piping",
        "action": "Lowered wrench with rope; cleared perimeter below",
        "text": "Mechanic working on 12-foot catwalk replacing 2-inch flange nuts dropped a 15-inch adjustable spanner wrench. The wrench slipped through the 1.5-inch toe-board gap on the open-grate mezzanine and plummeted to the ground-level walkway adjacent to the water treatment laboratory. No workers were directly below at the moment, but the walkway is a designated primary egress route without active barricades or drop netting.",
        "hazard": "Unsecured Tool Dropped from Mezzanine Walkway",
        "cat": HazardCategory.FALLING_OBJECTS_HEIGHT,
        "precursors": ["Non-tethered hand tools at elevation", "Substandard toe-board gap on mezzanine grating", "Absence of red-zone barricading below"],
        "assets": ["Ground-level personnel", "Water Lab Entrance"],
        "failed_safeguards": ["Tool lanyards/tethers", "Dropped object perimeter barricading"],
        "mitigation": "Mandate tool lanyards for all work > 4 feet; install mesh under-decking and toe-board skirting.",
        "risk": RiskLevel.MEDIUM,
        "score": 6.8,
        "rationale": "Heavy metal tool dropped 12 feet into active walkway. Potential for severe blunt-force trauma had a worker traversed below.",
        "citations": ["OSHA 1910.28 (Duty to have Fall Protection and Falling Object Protection)"],
        "signals": ["Dropped Object Precursor", "Working at Heights", "Missing Drop Zone Control"],
        "escalate": True
    },
    {
        "dept": "Chemical Processing & Synthesis",
        "loc": "Bulk Solvent Storage Tank Farm - Dike B",
        "role": "Offloading Specialist",
        "equip": "Toluene Tanker Truck Tank-T04",
        "env": "Outdoor transfer, sunny, dry, high static potential",
        "action": "Suspended transfer immediately and clamped grounding cable to certified copper busbar",
        "text": "Prior to starting bulk unloading of 5,000 gallons of toluene from rail car to Tank 4, operator noticed ground-proving monitor light was flickering amber instead of steady green. Visual inspection revealed ground clamp was attached over thick layer of rusted paint on truck chassis rather than bare metal earth point. Static charge during solvent transfer could have sparked flammable vapor ignition at vent port.",
        "hazard": "Improper Static Grounding During Flammable Solvent Offloading",
        "cat": HazardCategory.THERMAL_FIRE,
        "precursors": ["Grounding clamp clipped to insulated/painted metal", "Failure to verify zero-ohm continuity prior to hose connection"],
        "assets": ["Tanker Truck", "Bulk Farm Tank 4", "Transfer Depot"],
        "failed_safeguards": ["Automated pump interlock bypass", "Standard grounding continuity check SOP"],
        "mitigation": "Install intrinsically safe interlocked grounding clamps that inhibit pump startup until continuity <= 10 ohms.",
        "risk": RiskLevel.HIGH,
        "score": 9.4,
        "rationale": "Class IB flammable liquid transfer with ineffective static grounding. Direct precursor to major fire/explosion and mass casualty.",
        "citations": ["OSHA 1910.106 (Flammable Liquids)", "NFPA 77 (Recommended Practice on Static Electricity)"],
        "signals": ["Explosion Precursor", "Static Discharge Risk", "Flammable Vapor Zone"],
        "escalate": True
    },
    {
        "dept": "Semiconductor Cleanroom Assembly",
        "loc": "Photolithography Bay 6",
        "role": "Litho Technician",
        "equip": "Stepper Track Coat Unit 2",
        "env": "Class 100 cleanroom, yellow filtered lighting",
        "action": "Mopped puddle with certified cleanroom chemical pad and flagged facilities",
        "text": "Small puddle of isopropyl alcohol (IPA) approximately 100ml found pooled on conductive vinyl floor near solvent drain line. Operator's ESD shoe slipped slightly during wafer cassette transfer, but balance was maintained with handrail. Drain tubing connection had loosened due to thermal cycling from exhaust duct.",
        "hazard": "Solvent Puddle Causing Minor Slip Hazard in Cleanroom",
        "cat": HazardCategory.SLIP_TRIP_FALL,
        "precursors": ["Loose solvent drain fitting", "Lack of secondary drip tray under coupling"],
        "assets": ["Litho Technician", "Wafer Cassette ($85k)"],
        "failed_safeguards": ["Secondary containment tray", "Preventative maintenance connection torque checks"],
        "mitigation": "Replace slip-fit tubing with compression fittings; add drip pan with liquid sensor.",
        "risk": RiskLevel.LOW,
        "score": 3.4,
        "rationale": "Minor liquid volume, no skin contact or fall occurred, low volatility hazard under high-exchange cleanroom airflow.",
        "citations": ["OSHA 1910.22 (General Walking-Working Surfaces)"],
        "signals": ["Slip Hazard", "Minor Leak"],
        "escalate": False
    },
    {
        "dept": "Heavy Fabrication & Stamping",
        "loc": "Robotic Welding Cell 4",
        "role": "Welding Inspector",
        "equip": "Fanuc 6-Axis Arc Mate Robot",
        "env": "Darkened arc curtain enclosure, ambient shop noise",
        "action": "Exited cell and reported safety circuit bypass to engineering lead",
        "text": "Maintenance technician entered robot enclosure to clear weld wire bird-nest while interlock access gate was keyed open with an override magnetic bypass wedge. Robot was paused in hold mode rather than zero-energy state (LOTO). While mechanic was untangling wire nozzle, teach pendant was nudged on console table, causing robot arm to jog 15cm toward mechanic's shoulder before halting on torque limit.",
        "hazard": "Unauthorized Defeat of Robot Cell Safety Interlock During Live Clearing",
        "cat": HazardCategory.MECHANICAL_CRUSH,
        "precursors": ["Magnetic bypass key used on safety gate", "Failure to isolate energy via Lockout/Tagout (LOTO)", "Teach pendant left in active mode"],
        "assets": ["Maintenance Technician", "Robotic Arm"],
        "failed_safeguards": ["Dual-channel gate interlock switch", "Lockout/Tagout policy (LOTO)"],
        "mitigation": "Confiscate magnetic bypass shunts; institute trapped-key interlock system (Kirk key) requiring physical lock removal before entry.",
        "risk": RiskLevel.HIGH,
        "score": 9.5,
        "rationale": "Intentional bypass of safety interlock in automated machinery space. High-severity precursor to crushing or struck-by fatality.",
        "citations": ["OSHA 1910.147 (The Control of Hazardous Energy - Lockout/Tagout)", "ANSI/RIA R15.06 (Industrial Robots Safety)"],
        "signals": ["Fatal Precursor", "LOTO Violation", "Interlock Tampering"],
        "escalate": True
    },
    {
        "dept": "Fleet Loading Dock & Yard Operations",
        "loc": "Dock Door 8 - Trailer Staging",
        "role": "Material Handler",
        "equip": "Dock Leveler DL-08 & 53ft Dry Van Trailer",
        "env": "Heavy rain, wet concrete yard ramp",
        "action": "Halted forklift entry; engaged manual steel wheel chocks and re-hooked trailer restraint",
        "text": "Pallet truck driver drove front wheels onto dock plate inside trailer when trailer creep occurred, pulling trailer 6 inches away from dock bumper. The automated vehicle restraint (dock lock) had failed to engage due to accumulated mud on the trailer ICC bar, but the indoor dock control panel showed a false green 'Restrained' light. Forklift mast jarred violently across the expanding gap before driver reversed back onto concrete warehouse slab.",
        "hazard": "Trailer Creep with False Positive Dock Lock Sensor",
        "cat": HazardCategory.MECHANICAL_CRUSH,
        "precursors": ["Debris on trailer bumper bar blocking lock jaw", "Faulty microswitch indicator giving false safe signal", "Absence of secondary wheel chocks"],
        "assets": ["Pallet Truck Operator", "Dock Equipment"],
        "failed_safeguards": ["Automated trailer restraint system", "Secondary wheel chocking protocol"],
        "mitigation": "Service dock lock proximity sensors; enforce mandatory manual wheel chocks alongside mechanical lock.",
        "risk": RiskLevel.HIGH,
        "score": 9.2,
        "rationale": "Dock separation while powered mobile equipment is entering trailer. Extreme tip-over and catastrophic fall hazard.",
        "citations": ["OSHA 1910.178(k)(1) (Trucks and Railroad Cars Loading Safety)"],
        "signals": ["Dock Separation Precursor", "Sensor Failure", "Forklift Tip-Over Risk"],
        "escalate": True
    },
    {
        "dept": "Plant Facilities & Maintenance",
        "loc": "Substation B - 480V Motor Control Center (MCC)",
        "role": "Senior Electrician",
        "equip": "Cutler-Hammer 480V Bucket Switchgear",
        "env": "Indoor climate controlled electrical vault",
        "action": "Donned 40 cal/cm² arc flash suit and applied lock before resuming",
        "text": "Electrician was preparing to rack in a 480V breaker bucket after motor maintenance. Had removed dead-front panel and was reaching in with insulated socket without wearing NFPA 70E Arc Flash face shield and balaclava (only wearing 8 cal glasses). An ungrounded busbar had dropped a washer behind phase B during prior shift. Racking was paused when electrician noticed slight scorch mark on bus support.",
        "hazard": "Energized 480V Switchgear Racking without Arc Flash PPE",
        "cat": HazardCategory.ELECTRICAL,
        "precursors": ["Missing Cal/cm² arc-rated PPE", "Foreign conductive object in enclosure", "Absence of thermographic pre-check"],
        "assets": ["Senior Electrician", "MCC Substation"],
        "failed_safeguards": ["NFPA 70E PPE compliance", "Foreign material exclusion protocol"],
        "mitigation": "Mandate remote racking mechanisms; audit compliance on NFPA 70E PPE boundaries.",
        "risk": RiskLevel.HIGH,
        "score": 9.3,
        "rationale": "Severe arc flash potential in 480V high-energy enclosure with missing blast PPE. Direct fatal precursor.",
        "citations": ["OSHA 1910.303 (General Electrical Safety)", "NFPA 70E (Standard for Electrical Safety in the Workplace)"],
        "signals": ["Arc Flash Precursor", "Electrical Arc Blast Risk", "PPE Deficiency"],
        "escalate": True
    },
    {
        "dept": "High-Bay Warehousing & Logistics",
        "loc": "Packaging Staging Area - East Wall",
        "role": "Packer",
        "equip": "Manual Carton Sealing Machine",
        "env": "Normal warehouse lighting, dry",
        "action": "Cleaned tape debris and unplugged unit for maintenance",
        "text": "Operator was guiding corrugated boxes into automatic tape applicator when tape cutter blade stuck in extended position. Operator reached fingers within 1 inch of serrated spring-loaded blade to yank tangled packing tape. Guard cover hinge was broken and taped down with duct tape, removing physical barrier.",
        "hazard": "Exposed Serrated Cutter Blade on Packaging Machine",
        "cat": HazardCategory.MECHANICAL_CRUSH,
        "precursors": ["Defeated/broken finger guard held by duct tape", "Reaching into moving blade path to clear jam"],
        "assets": ["Packer"],
        "failed_safeguards": ["Interlocked finger blade guard", "Jam clearing tool/tongs"],
        "mitigation": "Replace blade guard with interlocked safety hood; provide non-conductive clearing picks.",
        "risk": RiskLevel.MEDIUM,
        "score": 6.2,
        "rationale": "Laceration hazard with missing machine guard. Moderate injury potential without bone/amputation severity.",
        "citations": ["OSHA 1910.212 (Machine Guarding)"],
        "signals": ["Guarding Defect", "Laceration Precursor"],
        "escalate": False
    },
    {
        "dept": "Chemical Processing & Synthesis",
        "loc": "Quality Control Wet Lab 2",
        "role": "Lab Chemist",
        "equip": "Fume Hood #4",
        "env": "Fume hood sash partially raised",
        "action": "Closed sash completely and recalibrated airflow monitor",
        "text": "Chemist was preparing aqua regia solution (nitric and hydrochloric acid) inside Fume Hood 4. Magnehelic differential pressure gauge showed low face velocity (55 FPM vs required 100 FPM), and low airflow alarm had been taped over with masking tape to silence the beeping. Faint pungent chlorine-like odor was detected outside sash before chemist noticed silenced alarm panel.",
        "hazard": "Compromised Fume Hood Face Velocity with Silenced Alarm During Toxic Acid Digestion",
        "cat": HazardCategory.ATMOSPHERIC_CONFINED,
        "precursors": ["Silenced/taped over safety acoustic alarm", "Inadequate face velocity ventilation", "Working with fuming toxic acid combination"],
        "assets": ["Lab Chemist", "QC Lab Staff"],
        "failed_safeguards": ["Exhaust airflow velocity threshold", "Audible alarm annunciator"],
        "mitigation": "Hardwire airflow alarms with supervisory tamper alarms; lock out hood until exhaust blower belt is serviced.",
        "risk": RiskLevel.HIGH,
        "score": 8.7,
        "rationale": "Toxic gas inhalation hazard combined with intentional silencing of primary safety warning device.",
        "citations": ["OSHA 1910.1450 (Occupational Exposure to Hazardous Chemicals in Laboratories)"],
        "signals": ["Toxic Gas Precursor", "Alarm Bypassed", "Chemical Exposure Hazard"],
        "escalate": True
    },
    {
        "dept": "Fleet Loading Dock & Yard Operations",
        "loc": "Trailer Parking Row D",
        "role": "Yard Hostler Driver",
        "equip": "Ottawa T2 Yard Jockey Truck #12",
        "env": "Night shift, heavy snow flurries, ice patches",
        "action": "Applied salt spreading and notified yard supervisor",
        "text": "While stepping down from the cab ladder of Yard Jockey #12, hostler slipped on black ice coating the lower steel rung. Driver grabbed the hand grab-rail with both gloved hands, preventing a backward fall onto the frozen asphalt. The cab exterior access light was burnt out, making the iced step invisible.",
        "hazard": "Iced Cab Step and Inoperative Exterior Step Light on Yard Truck",
        "cat": HazardCategory.SLIP_TRIP_FALL,
        "precursors": ["Burnt-out entry illumination bulb", "Ice accumulation on non-slip perforated rung"],
        "assets": ["Yard Hostler Driver"],
        "failed_safeguards": ["Three-point contact protocol (barely maintained)", "Yard winterization salting"],
        "mitigation": "Replace step lights with weatherproof LEDs; apply high-traction rubberized rung grips.",
        "risk": RiskLevel.LOW,
        "score": 3.8,
        "rationale": "Slip event without fall due to proper three-point contact maintained. Low potential for severe permanent harm.",
        "citations": ["OSHA 1910.22 (Walking-Working Surfaces)"],
        "signals": ["Slip Incident", "Poor Lighting"],
        "escalate": False
    },
    {
        "dept": "Heavy Fabrication & Stamping",
        "loc": "Overhead Crane Bay 1 - Coil Unloading Bay",
        "role": "Rigger / Slinger",
        "equip": "20-Ton Overhead Bridge Crane CR-01",
        "env": "Industrial shop floor, oily surface",
        "action": "Lowered coil back into cradle and tagged out synthetic sling",
        "text": "While rigging a 14,000 lb steel slit coil using a synthetic round sling, rigger noticed severe friction burn and outer jacket fraying exposing inner core yarn strands on the sling choker point. Corner softeners (cut protectors) had not been placed over the sharp sheared 90-degree steel edges of the coil. Sling was under 70% rated tension when noticed.",
        "hazard": "Damaged Synthetic Lifting Sling Rigged Over Unprotected Sharp Steel Edge",
        "cat": HazardCategory.FALLING_OBJECTS_HEIGHT,
        "precursors": ["Missing corner edge protectors/softeners", "Severely frayed outer protective cover", "Heavy suspended dynamic load"],
        "assets": ["Rigger", "Coil Unloading Crew"],
        "failed_safeguards": ["Pre-lift sling inspection", "Edge protection protocol"],
        "mitigation": "Destroy and dispose damaged sling; mandate magnetic polyurethane corner protectors on all coil lifts.",
        "risk": RiskLevel.HIGH,
        "score": 9.4,
        "rationale": "Suspended multi-ton load with degraded rigging over sharp edge. Direct precursor to catastrophic load drop and crushing fatality.",
        "citations": ["OSHA 1910.184 (Slings)", "ASME B30.9 (Slings and Rigging Standards)"],
        "signals": ["Fatal Precursor", "Rigging Failure Risk", "Suspended Heavy Load"],
        "escalate": True
    },
    {
        "dept": "Plant Facilities & Maintenance",
        "loc": "Waste Water Treatment Basin 2",
        "role": "Environmental Tech",
        "equip": "Submersible Aerator Pump AP-02",
        "env": "Humid, sewage odor, confined basin pit",
        "action": "Refused entry, recalibrated 4-gas meter, and ventilated pit for 30 minutes",
        "text": "Technician preparing to enter 10-foot deep aeration basin pit to unbolt pump flange conducted atmospheric test with 4-gas monitor through manhole hatch. Initial reading showed H2S at 4.2 PPM (action level 5 PPM). However, technician noticed calibration sticker had expired 8 months ago and bump test gas canister was completely empty. Secondary backup calibrated detector measured H2S at 18.5 PPM (exceeding OSHA PEL of 10 PPM ceiling).",
        "hazard": "Confined Space Entry Attempt with Out-of-Calibration Gas Detector in High H2S Environment",
        "cat": HazardCategory.ATMOSPHERIC_CONFINED,
        "precursors": ["Expired instrument calibration sticker", "Empty bump test verification station", "Elevated toxic H2S atmosphere in recessed basin"],
        "assets": ["Environmental Tech", "Entry Attendant"],
        "failed_safeguards": ["Pre-entry instrument bump test protocol", "Calibration tracking database"],
        "mitigation": "Lock out uncalibrated detectors with automated dock test docking stations; retrain on confined space entry permit verification.",
        "risk": RiskLevel.HIGH,
        "score": 9.7,
        "rationale": "Entry into deadly toxic gas atmosphere with compromised monitoring equipment. Premier fatal precursor in industrial operations.",
        "citations": ["OSHA 1910.146 (Permit-Required Confined Spaces)"],
        "signals": ["Fatal Precursor", "Confined Space Hazard", "H2S Gas Toxicity", "Faulty Gas Monitor"],
        "escalate": True
    },
    {
        "dept": "Semiconductor Cleanroom Assembly",
        "loc": "Chemical Mechanical Polishing (CMP) Slurry Room",
        "role": "Process Engineer",
        "equip": "Silica Slurry Day Tank T-301",
        "env": "Damp floor, white protective bunny suit",
        "action": "Tightened union fitting and placed absorbent boom around spill",
        "text": "During filter replacement on colloidal silica polishing slurry line, small drip (approx 50 drops/min) developed at quick-disconnect coupling. Slurry dried into extremely slick glassy residue on epoxy flooring. Operator experienced minor foot slip while stepping over pipe bridge, held railing.",
        "hazard": "Colloidal Slurry Drip Creating High Slip Floor Condition",
        "cat": HazardCategory.SLIP_TRIP_FALL,
        "precursors": ["Worn Viton O-ring in quick-disconnect coupling", "Dried silica residue forming slippery film"],
        "assets": ["Process Engineer"],
        "failed_safeguards": ["Drip collection tray", "O-ring scheduled PM"],
        "mitigation": "Replace couplings with double-shutoff dry-break fittings; install drip catch pans.",
        "risk": RiskLevel.LOW,
        "score": 3.2,
        "rationale": "Low chemical toxicity (pH neutral silica), minor slip without fall, fast cleanup.",
        "citations": ["OSHA 1910.22 (Walking-Working Surfaces)"],
        "signals": ["Slip Risk", "Minor Chemical Leak"],
        "escalate": False
    },
]


def generate_synthetic_reports(target_count: int = 42) -> List[EnrichedReport]:
    """Generates an enriched dataset of realistic industrial near-miss reports."""
    reports: List[EnrichedReport] = []
    
    # Base real scenarios
    for idx, base in enumerate(RAW_REPORTS_DATA):
        report_id = f"NM-2024-{idx+1:03d}"
        days_ago = random.randint(1, 90)
        report_date = (datetime.now() - timedelta(days=days_ago, hours=random.randint(1, 23))).strftime("%Y-%m-%d %H:%M:%S")
        facility = FACILITIES[idx % len(FACILITIES)]
        
        raw_report = NearMissReport(
            id=report_id,
            timestamp=report_date,
            facility=facility,
            department=base["dept"],
            location_specific=base["loc"],
            reporter_role=base["role"],
            raw_text=base["text"],
            equipment_involved=base["equip"],
            environmental_factors=base["env"],
            immediate_action_taken=base["action"],
        )
        
        extraction = StructuredExtraction(
            report_id=report_id,
            primary_hazard=base["hazard"],
            hazard_category=base["cat"],
            precursor_events=base["precursors"],
            affected_assets=base["assets"],
            failed_safeguards=base["failed_safeguards"],
            recommended_mitigation=base["mitigation"],
        )
        
        assessment = RiskAssessment(
            report_id=report_id,
            risk_level=base["risk"],
            risk_score=base["score"],
            confidence=round(random.uniform(0.88, 0.98), 2),
            rationale=base["rationale"],
            osha_citations=base["citations"],
            precursor_severity_signals=base["signals"],
            escalation_potential=base["escalate"],
        )
        
        enriched = EnrichedReport(
            report=raw_report,
            extraction=extraction,
            assessment=assessment,
            overrides=[],
            effective_risk_level=assessment.risk_level,
        )
        reports.append(enriched)

    # Parametric generation to reach target_count (expanding across diverse variations)
    templates = [
        {
            "dept": "High-Bay Warehousing & Logistics",
            "loc_prefix": "Pallet Racking Bay",
            "role": "Reach Truck Driver",
            "hazard_tmpl": "Overloaded Wire Mesh Decking on High-Tier Pallet Rack",
            "cat": HazardCategory.FALLING_OBJECTS_HEIGHT,
            "text_tmpl": "Operator observed heavy steel pallet (weight tag: 3,200 lbs) placed onto wire mesh deck rated for 2,500 lbs at rack level 4 (24 feet elevation). Wire decking had visible downward deflection of 1.5 inches and corner weld was cracked. Pallet had begun tilting slightly towards pedestrian picking lane.",
            "equip": "Crown Reach Truck RR-5700 & Tier-4 Racking",
            "precursors": ["Exceeded rack shelf weight rating", "Cracked wire deck weld", "Improper weight tag auditing"],
            "safeguards": ["Rack load rating signage", "Forklift scale verification"],
            "risk": RiskLevel.HIGH, "score": 8.8, "escalate": True,
            "citations": ["OSHA 1910.176 (Handling Materials - General)"],
            "signals": ["Falling Object Precursor", "Structural Racking Defect"]
        },
        {
            "dept": "Heavy Fabrication & Stamping",
            "loc_prefix": "Structural Beam Welding Bay",
            "role": "Structural Welder",
            "hazard_tmpl": "Damaged Ground Clamp Cable Causing Arc Strike on Pressurized Gas Line",
            "cat": HazardCategory.ELECTRICAL,
            "text_tmpl": "Welder initiated SMAW arc on I-beam fixture. Frayed copper ground cable with split rubber casing made contact with adjacent 100 PSI compressed air supply line, creating a localized high-amp flash and burning pinhole through the air manifold hose.",
            "equip": "Miller 400A Mig/Tig Welder",
            "precursors": ["Split insulation on welding ground return lead", "Routing welding leads across gas manifolds"],
            "safeguards": ["Pre-work hot work lead inspection", "Separation of electrical and pneumatic lines"],
            "risk": RiskLevel.MEDIUM, "score": 6.5, "escalate": False,
            "citations": ["OSHA 1910.252 (General Requirements - Welding, Cutting and Brazing)"],
            "signals": ["Arc Strike Precursor", "Damaged Cable Insulation"]
        },
        {
            "dept": "Chemical Processing & Synthesis",
            "loc_prefix": "Centrifuge Room 1",
            "role": "Process Tech",
            "equip": "Basket Centrifuge CF-101",
            "hazard_tmpl": "Nitrogen Blanketing Depressurization on Flammable Cake Discharge",
            "cat": HazardCategory.PROCESS_SAFETY,
            "text_tmpl": "During solvent-wetted solids discharge from basket centrifuge, nitrogen inerting purge valve pressure dipped below 0.5 inches water column setpoint. Low N2 pressure interlock failed to trip drive motor, allowing spinning basket to continue discharging in presence of ambient oxygen and flammable cyclohexane vapors.",
            "precursors": ["Nitrogen supply pressure excursion", "Failed safety interlock trip circuit", "Flammable solvent vapor concentration"],
            "safeguards": ["Inert gas interlock switch", "LEL vapor sensor"],
            "risk": RiskLevel.HIGH, "score": 9.5, "escalate": True,
            "citations": ["OSHA 1910.119 (Process Safety Management of Highly Hazardous Chemicals)", "NFPA 69 (Explosion Prevention Systems)"],
            "signals": ["Explosion Precursor", "PSM Critical Failure", "Loss of Inerting"]
        },
        {
            "dept": "Plant Facilities & Maintenance",
            "loc_prefix": "Chiller Plant Roof Section",
            "role": "Roofer / Facility Tech",
            "equip": "Cooling Tower CT-03",
            "hazard_tmpl": "Missing Parapet Fall Arrest Anchor on High Roof Edge",
            "cat": HazardCategory.FALLING_OBJECTS_HEIGHT,
            "text_tmpl": "Technician walked to edge of roof 35 feet above ground to inspect cooling tower overflow pipe. Warning flag line was missing within 6 feet of leading edge, and permanent D-ring roof anchor was heavily rusted with broken swage sleeve.",
            "precursors": ["Corroded fall arrest anchor point", "Missing roof warning perimeter demarcations"],
            "safeguards": ["Engineered fall protection anchor", "Visual perimeter warning line"],
            "risk": RiskLevel.HIGH, "score": 8.9, "escalate": True,
            "citations": ["OSHA 1910.28 (Fall Protection Systems)"],
            "signals": ["Fall from Height Precursor", "Defective Anchor Point"]
        },
        {
            "dept": "Fleet Loading Dock & Yard Operations",
            "loc_prefix": "Fuel Island Station 2",
            "role": "Yard Mechanic",
            "equip": "Diesel Dispenser Pump D-02",
            "hazard_tmpl": "Emergency Fuel Cutoff Switch Jammed in Manual Position",
            "cat": HazardCategory.THERMAL_FIRE,
            "text_tmpl": "Routine quarterly testing of red mushroom emergency fuel shutoff switch at diesel island revealed internal mechanical spring had corroded and button remained stuck in 'Run' position despite full manual depression.",
            "precursors": ["Corroded mechanical trip spring", "Infrequent emergency switch exercising"],
            "safeguards": ["Emergency fuel shutoff (E-Stop)", "Monthly functional testing"],
            "risk": RiskLevel.MEDIUM, "score": 5.9, "escalate": False,
            "citations": ["OSHA 1910.106 (Flammable and Combustible Liquids)"],
            "signals": ["E-Stop Failure", "Fire Suppression Precursor"]
        },
        {
            "dept": "Semiconductor Cleanroom Assembly",
            "loc_prefix": "Wet Etch Bench WB-04",
            "role": "Etch Specialist",
            "equip": "Hydrofluoric Acid Bath Unit",
            "hazard_tmpl": "Expired Calcium Gluconate Antidote Kit at HF Acid Station",
            "cat": HazardCategory.CHEMICAL_TOXIC,
            "text_tmpl": "During weekly first-aid audit at hydrofluoric acid (HF) wet bench, operator discovered emergency calcium gluconate gel tube in the wall dispenser box was expired by 14 months and crusty at seal. Eye-wash bottle seal had also been broken.",
            "precursors": ["Expired medical antidote in critical chemical zone", "Unsealed secondary emergency eye wash"],
            "safeguards": ["Medical emergency antidote availability", "Weekly safety equipment inspection checklist"],
            "risk": RiskLevel.MEDIUM, "score": 6.7, "escalate": True,
            "citations": ["OSHA 1910.1200", "OSHA 1910.151 (Medical Services and First Aid)"],
            "signals": ["Chemical Safety Failure", "Emergency Readiness Deficit"]
        },
        {
            "dept": "High-Bay Warehousing & Logistics",
            "loc_prefix": "Battery Charging Station B",
            "role": "Battery Tech",
            "equip": "Lead-Acid Battery Changer Crane",
            "hazard_tmpl": "Defective Eyewash Eyecup Missing Dust Caps in Acid Battery Room",
            "cat": HazardCategory.CHEMICAL_TOXIC,
            "text_tmpl": "Technician checking forklift battery electrolyte levels noticed plumbed emergency eye-wash station nozzles had accumulated battery charging acid-mist residue due to missing yellow pop-off dust caps. Testing station sprayed discolored water for first 6 seconds.",
            "precursors": ["Missing dust caps on emergency eyewash", "Acid mist deposition on plumbing orifices"],
            "safeguards": ["ANSI Z358.1 Weekly flush protocol", "Protective dust caps"],
            "risk": RiskLevel.LOW, "score": 4.1, "escalate": False,
            "citations": ["OSHA 1910.151(c) (Eyewash Facilities)", "ANSI Z358.1"],
            "signals": ["First Aid Defect", "Maintenance Lag"]
        },
        {
            "dept": "Heavy Fabrication & Stamping",
            "loc_prefix": "Plasma CNC Cutting Table 3",
            "role": "CNC Operator",
            "equip": "Koike 5-Axis CNC Plasma Burner",
            "hazard_tmpl": "Water Table Sludge Build-up Blocking Emergency Slag Extraction",
            "cat": HazardCategory.THERMAL_FIRE,
            "text_tmpl": "During heavy plate plasma beveling, molten slag accumulated on dry scrap plate edge due to low water table level. Minor flame flare ignited pallet wooden base stored 2 feet from table edge. Flame extinguished with CO2 extinguisher in under 10 seconds.",
            "precursors": ["Low water table level", "Combustible wood pallet stored inside 5-foot hot work perimeter"],
            "safeguards": ["35-foot hot work combustible clearance standard", "Water quench table"],
            "risk": RiskLevel.LOW, "score": 3.9, "escalate": False,
            "citations": ["OSHA 1910.252(a) (Fire Prevention in Welding)"],
            "signals": ["Hot Work Violation", "Housekeeping Hazard"]
        }
    ]

    curr_idx = len(reports) + 1
    while len(reports) < target_count:
        tmpl = templates[len(reports) % len(templates)]
        report_id = f"NM-2024-{curr_idx:03d}"
        days_ago = random.randint(2, 120)
        report_date = (datetime.now() - timedelta(days=days_ago, hours=random.randint(1, 23))).strftime("%Y-%m-%d %H:%M:%S")
        facility = FACILITIES[curr_idx % len(FACILITIES)]
        loc = f"{tmpl['loc_prefix']} #{random.randint(1, 12)}"

        raw_report = NearMissReport(
            id=report_id,
            timestamp=report_date,
            facility=facility,
            department=tmpl["dept"],
            location_specific=loc,
            reporter_role=tmpl["role"],
            raw_text=tmpl["text_tmpl"],
            equipment_involved=tmpl["equip"],
            environmental_factors="Standard factory conditions",
            immediate_action_taken="Safety officer logged issue, corrective action ticket generated",
        )

        extraction = StructuredExtraction(
            report_id=report_id,
            primary_hazard=tmpl["hazard_tmpl"],
            hazard_category=tmpl["cat"],
            precursor_events=tmpl["precursors"],
            affected_assets=["Operating Personnel", tmpl["equip"]],
            failed_safeguards=tmpl["safeguards"],
            recommended_mitigation="Audit preventive maintenance and implement physical safeguards.",
        )

        assessment = RiskAssessment(
            report_id=report_id,
            risk_level=tmpl["risk"],
            risk_score=tmpl["score"],
            confidence=round(random.uniform(0.89, 0.97), 2),
            rationale=f"Automated risk scoring based on precursor pattern evaluation for {tmpl['cat'].value}.",
            osha_citations=tmpl["citations"],
            precursor_severity_signals=tmpl["signals"],
            escalation_potential=tmpl["escalate"],
        )

        enriched = EnrichedReport(
            report=raw_report,
            extraction=extraction,
            assessment=assessment,
            overrides=[],
            effective_risk_level=assessment.risk_level,
        )
        reports.append(enriched)
        curr_idx += 1

    return reports
