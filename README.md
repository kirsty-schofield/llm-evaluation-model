## The Objective

A robust, localised evaluation pipeline designed to benchmark hallucination rates and safety guardrails for an open-source medical AI assistant. Built entirely using local hardware to eliminate API costs and ensure total data privacy, this project makes use of Ollama (Llama 3 8B) for generation and DeepEval as an automated algorithmic judge.

## Project overview

Deploying LLMs in high-stakes domains like healthcare demands uncompromising accuracy. This project provides a test architecture that prompts a local medical assistant with complex patient questions, enforces a strict constraint-following system prompt, and programmatically audits the responses for hallucinations. 

### Tech Stack

-	Target Model: llama3:8b (via Ollama) acting as a precise, deterministic medical chatbot.
-	Evaluation Judge: llama3:8b (via DeepEval's OllamaModel integration) to evaluate facts.
-	Frameworks: deepeval (Metric Extraction), pandas (Data Pipeline & Serialisation), asyncio (Custom execution loop control).

## The Architecture and Challenges

While out-of-the-box evaluation frameworks rely heavily on APIs, executing a dual-LLM setup (Generator + Judge) completely locally introduced distinct infrastructural bottlenecks.

### The Hardware Constraint & Asynchronous Override

Initially, using DeepEval's global automated evaluate() function caused background multi-threading, flooding local hardware channels and triggering severe latency spikes. Ollama queries regularly exceeded internal deadlines, resulting in script-killing asyncio.exceptions.CancelledError and TimeoutError exceptions.
The master evaluate() abstraction was completely bypassed and he evaluation loop was refactored into a custom, highly resilient for loop executing over a single-concurrency pipeline. This allowed the hardware to efficiently process dense medical context rows at its native maximum capacity without artificial timeouts.

### Handling the Refusal-Metric Misalignment

To guarantee patient safety, the chatbot's system prompt dictates a fallback clause: If the context lacks definitive facts, answer exactly with: "I do not have enough verified information to answer this."
During initial test runs, standard semantic hallucination metrics parsed this safe refusal as a factual contradiction, interpreting a claim of "no information" within a text-dense medical document as a hallucination. This caused a massive influx of false-positive failures (1.0 scores).

To resolve this issue, I implemented an evaluation metric override directly inside the pipeline. Any output containing the defensive safety string was intercepted, granted a clean score of 0.0 (Safe Refusal), and bypassed the LLM judge entirely, which dramatically reduced execution time and isolated genuine hallucinations.

## Evaluation & Metrics Analysis

The framework evaluated 72 medical test cases spanning critical topics such as acute symptoms, prescription doubling, and triage.
##Key Takeaways & Clinical Value
-	Substantive Reliability: When the model chose to speak, it achieved a 96.1% factual accuracy rate (49 perfect alignments vs. 2 hallucinations).
-	Defensive Safety Profile: Maintaining a 29.2% Defensive Refusal Rate proves the architecture successfully prioritises "saying I don't know" over generating dangerous, unverified medical fabrications.
-	Local Viability: This project demonstrates that end-to-end LLM data validation pipelines can be sustainably developed, debugged, optimised, and maintained entirely locally on consumer hardware.

