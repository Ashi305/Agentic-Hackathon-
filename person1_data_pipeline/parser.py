import os
import PyPDF2
from schema import NearMissReport  
from storage import save_enriched_report   

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a given PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    
    return text.strip()

def parse_incident_text(raw_text, filename):
    """
    Structures the raw text into a format matching the pipeline's schema.
    For robust extraction (like pulling exact dates or hazard types), 
    you can integrate regex or a lightweight LLM call here.
    """
    structured_data = {
        "source_file": filename,
        "source_type": "PDF Report",
        "raw_description": raw_text,
        # Truncate for a summary field if your schema requires it
        "summary": raw_text[:200] + "..." if len(raw_text) > 200 else raw_text,
        "status": "pending_analysis"
    }
    return structured_data

def process_pdf_directory(directory_path):
    """Iterates through a folder of PDFs, extracts data, and sends to storage."""
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} created. Add PDFs and run again.")
        os.makedirs(directory_path)
        return

    processed_count = 0
    for filename in os.listdir(directory_path):
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
            
            # 3. Store (Uncomment and adapt to your actual storage.py functions)
            # report_object = NearMissReport(**parsed_data)
            # save_enriched_report(report_object)
            
            print(f"Successfully processed: {filename}")
            processed_count += 1
            
    print(f"\nFinished parsing {processed_count} PDF(s).")

if __name__ == "__main__":
    # Define where the PDFs will be stored (e.g., inside the data folder)
    target_directory = os.path.join(os.path.dirname(__file__), "data", "pdf_reports")
    process_pdf_directory(target_directory)