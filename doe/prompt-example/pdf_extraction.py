import fitz  # PyMuPDF
import json

def extract_pdf_text_to_json(pdf_path, output_json_path):
    """Extract text from each page of a PDF and save it as a JSON array."""
    extracted_pages = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                page_text = page.get_text()
                extracted_pages.append({
                    "page_number": page_number,
                    "text": page_text.strip()
                })

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_pages, f, indent=4, ensure_ascii=False)

        print(f"Text extracted to: {output_json_path}")
    except Exception as e:
        print(f"Failed to extract PDF: {e}")

# Example usage
if __name__ == "__main__":
    extract_pdf_text_to_json("../../pair-pair-dataset/ODIN.pdf", "../../pair-pair-dataset/ODIN.json")
