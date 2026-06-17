import os
import pandas as pd
from ollama import Client 
import sys
import subprocess

# Ensure required libraries are ready
subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama", "deepeval"])

from deepeval.models import OllamaModel
from deepeval.metrics import HallucinationMetric 
from deepeval.test_case import LLMTestCase 

# Initialise local chatbot engine client (Llama3 8B)
ollama_client = Client(host="http://localhost:11434")

# Configure local DeepEval Judge model
print("Configuring local DeepEval Judge model")
local_judge = OllamaModel(
    model="llama3:8b", 
    base_url="http://localhost:11434"
) 

# Instantiate the metric using local judge
hallucination_metric = HallucinationMetric(threshold=0.5, model=local_judge)

def get_model_output(query, context):
    """Generates an answer from your custom local medical chatbot."""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise, strict medical AI assistant. "
                    "Your task is to answer the user's question using ONLY the facts present in the Context. "
                    "Do not extrapolate, do not assume, and do not use your own external knowledge. "
                    "If the Context does not explicitly contain the answer, reply exactly with: "
                    "'I do not have enough verified information to answer this.'\n\n"
                    f"CONTEXT:\n{context}"
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        output = ollama_client.chat(
            model="llama3:8b", 
            messages=messages, 
            options={"temperature": 0.0} # Deterministic responses
        )
        return output.message.content
    
    except Exception as e:
        print(f"Error generating text: {e}")
        return "Error generating response."

# Load the prepared dataset
csv_file = "Medical_QA_Data_Ready.csv" 
df = pd.read_csv(csv_file).dropna(subset=['prompt', 'source_context']) # Clear NaNs

# Storage arrays
chatbot_outputs = []
test_cases = []

print(f"\n Generating answers for {len(df)} prompts sequentially...")

for index, row in df.iterrows():
    actual_output = get_model_output(row['prompt'], row['source_context'])
    
    # Save the text output so it can be added to the CSV later
    chatbot_outputs.append(actual_output)
    
    # Bundle into a DeepEval test case
    test_cases.append(LLMTestCase(
        input=row['prompt'],
        actual_output=actual_output,
        context=[row['source_context']]
    ))

# Add the outputs column directly back to the dataframe
df['chatbot_output'] = chatbot_outputs

# Sequential evaluation

print(f"\n Starting Evaluation Phase on {len(test_cases)} rows.")

hallucination_scores = []
hallucination_reasons = []

# Define chatbot's exact safety fallback string
FALLBACK_STRING = "I do not have enough verified information to answer this."

for i, test_case in enumerate(test_cases):
    print(f"Evaluating row {i+1}/{len(test_cases)}: '{test_case.input[:40]}'")
    
    # Check if the chatbot output is a safe refusal fallback
    if FALLBACK_STRING.lower() in test_case.actual_output.lower():
        print("Safe Refusal detected. Bypassing judge.")
        score = 0.0
        reason = "The chatbot safely declined to answer. No hallucination occurred."
        
    else:
        # If it's an important answer, pass it to DeepEval to inspect for actual lies
        try:
            hallucination_metric.measure(test_case)
            score = hallucination_metric.score
            reason = hallucination_metric.reason
            print(f"Score: {score}")
        except Exception as e:
            print(f"Row {i+1} failed with error: {e}. Logging as an evaluation error.")
            score = None
            reason = f"Evaluation failed due to engine error: {str(e)}"
        
    hallucination_scores.append(score)
    hallucination_reasons.append(reason)

# Attach results back onto master DataFrame
df['hallucination_score'] = hallucination_scores
df['hallucination_reason'] = hallucination_reasons

# Save output
print("\n Progress complete. Exporting metrics.")

output_file = "Evaluated_Medical_QA_Data.csv"
df.to_csv(output_file, index=False)

print(f"Evaluation data compiled and saved to: {output_file}")
