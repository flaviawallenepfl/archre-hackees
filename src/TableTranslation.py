import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import Completion, CompletionChoice, CompletionUsage

# Load environment variables
load_dotenv()

 
# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to read a file (CSV or XLSX) and return its content
def read_file(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV and XLSX are supported.")

# Function to generate natural language summary using the file data
def generate_summary(file_data):
    # Convert the table to a string to pass it to OpenAI API
    table_str = file_data.to_string(index=False)

    prompt = f"""
    You are assisting in preparing tabular data for retrieval-augmented generation (RAG) search within an underwriting or reinsurance context. Your task is to describe a given table in clear, natural language and add insights. The description should:
    1. Summarize what the table contains
    2. Explain what the data represents (e.g., PML values, inflation rates, etc.)
    3. Note and explain the key insights from the data
    4. Highlight how the table is useful for an underwriter or actuarial/reinsurance professional
    5. Avoid unnecessary formatting or data dumps — focus on clarity and retrieval relevance
    
    Now apply the same logic to the following table:
    {table_str}
    """

    # Request OpenAI to generate a summary
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.5
    )

    return response.choices[0].message.content.strip()

# Main execution function
def main(file_path):
    try:
        # Step 1: Read the file
        file_data = read_file(file_path)

        # Step 2: Generate a summary based on the file content
        summary = generate_summary(file_data)

        # Step 3: Create the output .md file path (same name, .md extension)
        base_name = os.path.splitext(file_path)[0]
        output_path = f"{base_name}.md"

        # Step 4: Inform whether the file exists
        if os.path.exists(output_path):
            print(f"Overwriting existing Markdown file: {output_path}")
        else:
            print(f"Creating new Markdown file: {output_path}")

        # Step 5: Write the summary to the .md file
        with open(output_path, 'w') as f:
            f.write("# Insights\n\n")
            f.write(summary)

        print("Markdown summary successfully written.")

    except Exception as e:
        print(f"An error occurred: {e}")

# If this file is executed directly, run the main function
if __name__ == "__main__":
    # File path to the input CSV or XLSX file
    file_path = '/Users/elvankonukseven/hackermode4/archre-hackees/data/economics/interest-rates/Interest_rate_discount_rate_china.csv'#os.path.join('data', 'data/economics/interest-rates/Interest_rate_discount_rate_china.csv')

    # Execute main function
    main(file_path)
