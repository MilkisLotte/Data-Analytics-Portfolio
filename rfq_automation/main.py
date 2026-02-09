import pandas as pd
import spacy
from pathlib import Path
import sys

# --- 1. SET UP PATHS (Using Pathlib) ---
# This identifies the folder where THIS script is saved
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Create folders if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- 2. LOAD NLP MODEL ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy model not found. Run: python -m spacy download en_core_web_sm")
    sys.exit(1)


# --- 3. EXTRACTION LOGIC ---
def extract_rfq_entities(text):
    """
    Research Pipeline: Uses spaCy's NER (Named Entity Recognition)
    to extract quantities (CARDINAL) and locations (GPE).
    """
    doc = nlp(text)

    # Extract entities using list comprehension
    quantities = [ent.text for ent in doc.ents if ent.label_ == "CARDINAL"]
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

    return {
        "Extracted_Quantity": quantities[0] if quantities else "N/A",
        "Target_Location": locations[0] if locations else "Unknown",
        "NLP_Confidence": "High" if quantities or locations else "Low"
    }


# --- 4. MAIN EXECUTION ---
def main():
    csv_input = DATA_DIR / "rfq_data.csv"

    if not csv_input.exists():
        print(f"File not found: {csv_input}. Please place your CSV in the /data folder.")
        return

    # Load messy data
    df = pd.read_csv(csv_input)
    print(f"Successfully loaded {len(df)} RFQ entries.")

    # Apply the NLP brain to the Customer Message column
    # Using lambda to expand the dictionary into new columns
    extraction_results = df['Customer_Message'].apply(lambda x: pd.Series(extract_rfq_entities(x)))

    # Concatenate original data with extracted insights
    final_df = pd.concat([df, extraction_results], axis=1)

    # Export structured data for the Prodigy Graphics team
    output_path = OUTPUT_DIR / "automated_rfq_results.csv"
    final_df.to_csv(output_path, index=False)

    print(f"--- SUCCESS ---")
    print(f"Processed data saved to: {output_path}")


if __name__ == "__main__":
    main()