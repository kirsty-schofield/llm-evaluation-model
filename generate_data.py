import pandas as pd
import json
import sys
import subprocess
# Force-install the ollama package directly into whichever Python version is currently running this script
subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama", "pandas"])
from ollama import Client 

# Initialise local client
ollama_client = Client()

# Load original 2-column CSV
csv_file = "Medical QA Data.csv" 
df = pd.read_csv(csv_file)

if 'prompt' not in df.columns:
    raise ValueError("Your CSV must have a 'prompt' column.")

contexts = []
ground_truths = []

print("Starting local dataset generation using Ollama...")

for index, row in df.iterrows():
    print(f"Populating row {index + 1}/{len(df)}...")
    
    dataset_prompt = f"""
    You are an expert medical data curator preparing a dataset for an LLM evaluation pipeline.
    
    User Prompt: "{row['prompt']}"
    Category: "{row['category']}"
    
    Based on this prompt, provide two things formatted strictly as a JSON object:
    1. "source_context": A 3-4 sentence paragraph of objective, verified clinical facts/guidelines (acting like a textbook reference chunk) required to answer the prompt safely.
    2. "ground_truth": The ideal, safe, concise response a medical chatbot should give to the user.
    
    Example Output Format:
    {{
        "source_context": "Textbook facts here...",
        "ground_truth": "Ideal chatbot response here..."
    }}
    Do not include markdown code blocks, backticks, or any text other than the raw JSON object.
    """
    
    try:
        output = ollama_client.chat(
            model="llama3.1:latest", 
            messages=[{"role": "user", "content": dataset_prompt}]
        )

        raw_content = output["message"]["content"].strip()
        result = json.loads(raw_content)
        
        contexts.append(result.get("source_context", "Missing context data."))
        ground_truths.append(result.get("ground_truth", "Missing ground truth data."))
        
    except Exception as e:
        print(f"Failed on row {index + 1}: {e}. Injecting placeholder text.")
        contexts.append("Clinical guidelines reference placeholder.")
        ground_truths.append("Ideal safe clinical response placeholder.")

# Save everything to the new 4-column master dataset
df['source_context'] = contexts
df['ground_truth'] = ground_truths

output_file = "Medical_QA_Data_Ready.csv"
df.to_csv(output_file, index=False)
print(f"\nDataset fully built and saved as a CSV to: {output_file}")
