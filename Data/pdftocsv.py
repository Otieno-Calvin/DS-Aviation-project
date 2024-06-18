import re
import pandas as pd
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to parse the extracted text
def parse_airplane_data(text):
    lines = text.strip().split('\n')
    processed_data = []
    for line in lines:
        # Match the pattern of model, make, excluding the ending code
        match = re.match(r'(.+?), ([A-Z ]+?) [A-Z0-9]+$', line)
        if match:
            model, make = match.groups()
            processed_data.append([model.strip(), make.strip()])
    return processed_data

# Path to the PDF
pdf_path = r"C:\Users\GUEST USER\Downloads\Part3-By Model Number_Name.pdf"  # Use the appropriate path for the PDF

# Extract text from the provided PDF
text = extract_text_from_pdf(pdf_path)

# Parse the airplane data
data = parse_airplane_data(text)

# Create a DataFrame and save to CSV
df = pd.DataFrame(data, columns=["Model", "Make"])
csv_path = "C:/Users/GUEST USER/Documents/Flatiron/DS Project/airplane_data.csv"
df.to_csv(csv_path, index=False)

# Load the CSV file
df = pd.read_csv('airplane_data.csv')

# Remove the last part of the "Make" column
df['Make'] = df['Make'].apply(lambda x: ' '.join(x.split()[:-1]))

# Save the modified DataFrame back to the CSV file
df.to_csv('airplane_data.csv', index=False)

print(f"CSV file saved to {csv_path}")
