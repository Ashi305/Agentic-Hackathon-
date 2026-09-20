"""
Person 1: Automated PDF Document Ingestion & Parser
Extracts raw narrative text and structured incident entities from industrial
PDF safety observation reports, near-miss incident memos, and audit documentation.
"""
import os
import sys
import io
import re
from typing import Dict, Any, Optional, Union, List

# Ensure repository root is in sys.path for direct script execution and package imports
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Resilient PDF reader import (supports both PyPDF2 and pypdf)
try:
    import PyPDF2
except ImportError:
    try:
        import pypdf as PyPDF2
    except ImportError:
        PyPDF2 = None

# Resilient schema & storage imports
try:
    from person1_data_pipeline.schema import NearMissReport
    from person1_data_pipeline.storage import save_enriched_report, SafetyStorage
except ImportError:
    try:
        from .schema import NearMissReport
        from .storage import save_enriched_report, SafetyStorage
    except ImportError:
        from schema import NearMissReport
        from storage import save_enriched_report, SafetyStorage


def extract_text_from_pdf(pdf_source: Union[str, os.PathLike, bytes, Any]) -> str:
    """
    Extracts raw text from a given PDF file path, raw bytes, or file-like buffer
    (e.g., Streamlit UploadedFile or io.BytesIO).
    """
    if PyPDF2 is None:
        raise ImportError("PyPDF2 or pypdf is required to parse PDF documents. Run `pip install pypdf`.")

    text = ""
    try:
        if isinstance(pdf_source, (str, os.PathLike)):
            with open(pdf_source, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        elif isinstance(pdf_source, bytes):
            stream = io.BytesIO(pdf_source)
            reader = PyPDF2.PdfReader(stream)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        else:
            # File-like object (e.g. Streamlit UploadedFile)
            reader = PyPDF2.PdfReader(pdf_source)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF source: {e}")

    return text.strip()


def parse_incident_text(raw_text: str, filename: str = "document.pdf") -> Dict[str, Any]:
    """
    Structures the raw text into a format matching the pipeline's schema.
    Performs heuristic entity extraction to identify department, location,
    equipment, and clean narrative from standardized safety forms or unstructured text.
    """
    cleaned_text = raw_text.strip()
    
    # 1. Regex extractions for standard report headers if present
    dept_match = re.search(r"(?:Department|Dept|Area Sector)[:\-]\s*([^\n\r]+)", cleaned_text, re.IGNORECASE)
    loc_match = re.search(r"(?:Location|Site|Specific Location|Zone)[:\-]\s*([^\n\r]+)", cleaned_text, re.IGNORECASE)
    equip_match = re.search(r"(?:Equipment|Machinery|Asset|Unit Involved)[:\-]\s*([^\n\r]+)", cleaned_text, re.IGNORECASE)
    action_match = re.search(r"(?:Immediate Action|Action Taken|Resolution)[:\-]\s*([^\n\r]+)", cleaned_text, re.IGNORECASE)
    role_match = re.search(r"(?:Reporter|Observer|Role|Filed By)[:\-]\s*([^\n\r]+)", cleaned_text, re.IGNORECASE)

    # Inferred defaults if not explicitly structured in text
    department = dept_match.group(1).strip() if dept_match else None
    location = loc_match.group(1).strip() if loc_match else None
    equipment = equip_match.group(1).strip() if equip_match else None
    immediate_action = action_match.group(1).strip() if action_match else "Area inspected and supervisor notified"
    reporter_role = role_match.group(1).strip() if role_match else "EHS Field Inspector"

    # Heuristic department keyword classification if unlabelled
    if not department:
        lower_text = cleaned_text.lower()
        if any(w in lower_text for w in ["acid", "chemical", "flange", "reactor", "pump", "solvent", "valve"]):
            department = "Chemical Processing & Synthesis"
        elif any(w in lower_text for w in ["forklift", "dock", "aisle", "pallet", "warehouse", "rack"]):
            department = "High-Bay Warehousing & Logistics"
        elif any(w in lower_text for w in ["robot", "welding", "stamping", "press", "conveyor", "crane"]):
            department = "Heavy Fabrication & Stamping"
        elif any(w in lower_text for w in ["breaker", "substation", "transformer", "boiler", "catwalk", "maintenance"]):
            department = "Plant Facilities & Maintenance"
        elif any(w in lower_text for w in ["cleanroom", "semiconductor", "wafer", "lithography"]):
            department = "Semiconductor Cleanroom Assembly"
        else:
            department = "General Industrial Operations"

    if not location:
        location = f"Zone derived from {os.path.basename(filename)}"

    if not equipment:
        equipment = "Industrial Machinery / Process Unit"

    structured_data = {
        "source_file": filename,
        "source_type": "PDF Report",
        "raw_description": cleaned_text,
        "summary": cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text,
        "department": department,
        "location_specific": location,
        "equipment_involved": equipment,
        "immediate_action_taken": immediate_action,
        "reporter_role": reporter_role,
        "status": "pending_analysis"
    }
    return structured_data


def parse_pdf_report(pdf_source: Union[str, os.PathLike, bytes, Any], filename: str = "document.pdf") -> Dict[str, Any]:
    """
    High-level end-to-end function: extracts raw text and parses it into structured dictionary.
    """
    raw_text = extract_text_from_pdf(pdf_source)
    if not raw_text:
        return {
            "source_file": filename,
            "source_type": "PDF Report",
            "raw_description": "",
            "summary": "Empty or unscannable PDF document",
            "department": "General Operations",
            "location_specific": "Unknown",
            "equipment_involved": "N/A",
            "status": "error_empty"
        }
    return parse_incident_text(raw_text, filename=filename)


def process_pdf_directory(directory_path: str) -> int:
    """Iterates through a folder of PDFs, extracts data, and sends to storage."""
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} created. Add PDFs and run again.")
        os.makedirs(directory_path, exist_ok=True)
        return 0

    processed_count = 0
    storage = SafetyStorage()

    for filename in sorted(os.listdir(directory_path)):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            print(f"Parsing {filename}...")
            
            # 1. Extract
            raw_text = extract_text_from_pdf(file_path)
            if not raw_text:
                print(f"Warning: No text extracted from {filename}. May be a scanned image.")
                continue
                
            # 2. Parse/Structure
            parsed_data = parse_incident_text(raw_text, filename)
            
            # 3. Store record in warehouse
            try:
                report_obj = NearMissReport(
                    id=f"PDF-{abs(hash(filename)) % 100000:05d}",
                    facility="Central Production Facility",
                    department=parsed_data.get("department", "General Operations"),
                    location_specific=parsed_data.get("location_specific", "Facility Floor"),
                    reporter_role=parsed_data.get("reporter_role", "Safety Observer"),
                    raw_text=parsed_data.get("raw_description", raw_text),
                    equipment_involved=parsed_data.get("equipment_involved", "N/A"),
                    immediate_action_taken=parsed_data.get("immediate_action_taken", "Noted by inspector"),
                )
                # Store raw report
                with storage._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO reports (
                            id, timestamp, facility, department, location_specific,
                            reporter_role, raw_text, equipment_involved, environmental_factors, immediate_action_taken
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        report_obj.id, report_obj.timestamp, report_obj.facility, report_obj.department,
                        report_obj.location_specific, report_obj.reporter_role, report_obj.raw_text,
                        report_obj.equipment_involved, report_obj.environmental_factors, report_obj.immediate_action_taken
                    ))
                    conn.commit()
            except Exception as ex:
                print(f"Note: Storing raw PDF record: {ex}")

            print(f"Successfully processed: {filename}")
            processed_count += 1
            
    print(f"\nFinished parsing {processed_count} PDF(s).")
    return processed_count


if __name__ == "__main__":
    # Define where the PDFs will be stored (inside the data folder)
    target_directory = os.path.join(os.path.dirname(__file__), "data", "pdf_reports")
    process_pdf_directory(target_directory)