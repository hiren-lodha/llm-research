import ollama
import csv
import json
import os
import logging
import sys
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from threading import Lock
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mumbai_llm.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configuration
JSON_FILE = 'questions.json'
OUTPUT_CSV = 'results/llm_results.csv'
MODELS = ['phi3:latest', 'gemma:7b', 'llama3:latest', 'deepseek-llm:latest']  # Reordered: small to big
MAX_WORKERS = min(2, os.cpu_count() or 1)  # Conservative
RETRY_ATTEMPTS = 3
RETRY_DELAY = 10
WARMUP_TIMEOUT = 30

def check_system_resources():
    """Check available memory to avoid overloading."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    if available_gb < 4:
        logging.warning(f"Low memory available ({available_gb:.2f} GB). May cause model loading issues.")
    return available_gb >= 4

def load_questions(json_file: str) -> List[Dict]:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = []
            for category, items in data['categorized_questions'].items():
                for item in items:
                    questions.append({
                        "id": item["id"],
                        "category": category,
                        "english": item["text_en"],
                        "hindi": item["text_hi"]
                    })
            logging.info(f"Loaded {len(questions)} questions from {json_file}")
            return questions
    except Exception as e:
        logging.error(f"Error loading questions: {str(e)}")
        sys.exit(1)

def warm_up_model(model: str, max_retries: int = RETRY_ATTEMPTS) -> bool:
    warmup_question = "What is Mumbai famous for? Answer in one sentence."
    for attempt in range(max_retries):
        try:
            res = ollama.generate(
                model=model,
                prompt=warmup_question,
                options={"num_predict": 20, "timeout": WARMUP_TIMEOUT}
            )
            if res.get("response"):
                logging.info(f"{model} warm-up successful. Response: {res['response'][:50]}...")
                return True
        except Exception as e:
            logging.warning(f"{model} warm-up failed ({attempt+1}/{max_retries}): {e}")
            time.sleep(RETRY_DELAY)
    logging.error(f"{model} failed to warm up after {max_retries} attempts.")
    return False

def query_ollama(model: str, prompt: str, retries: int = RETRY_ATTEMPTS) -> Optional[str]:
    options = {
        "num_ctx": 2048,
        "temperature": 0.3,
        "seed": 42,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_predict": 512
    }
    for attempt in range(retries):
        try:
            start_time = time.time()
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options=options
            )
            elapsed = time.time() - start_time
            logging.debug(f"{model} response time: {elapsed:.2f}s")
            return response['message']['content']
        except Exception as e:
            logging.warning(f"{model} query failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))
    return None

def process_question(model: str, question: Dict) -> Dict:
    try:
        prompt_en = f"""Please answer the following question in English.
Be concise and accurate in your response.
Question: {question['english']}"""

        prompt_hi = f"""कृपया निम्नलिखित प्रश्न का उत्तर हिंदी में दें।
उत्तर संक्षिप्त और सटीक दें।
प्रश्न: {question['hindi']}"""

        res_en = query_ollama(model, prompt_en)
        res_hi = query_ollama(model, prompt_hi)

        return {
            "id": question["id"],
            "category": question["category"],
            "english": question["english"],
            "hindi": question["hindi"],
            "model": model,
            "response_en": res_en or "ERROR",
            "response_hi": res_hi or "ERROR",
            "tokens_en": len(res_en.split()) if res_en else 0,
            "tokens_hi": len(res_hi.split()) if res_hi else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logging.error(f"Failed question {question['id']} for {model}: {e}")
        return {
            "id": question["id"],
            "category": question["category"],
            "english": question["english"],
            "hindi": question["hindi"],
            "model": model,
            "response_en": "ERROR",
            "response_hi": "ERROR",
            "tokens_en": 0,
            "tokens_hi": 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def process_model_questions(model: str, questions: List[Dict]):
    lock = Lock()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_question, model, question) for question in questions]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {model}"):
            result = future.result()
            with lock:
                with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        result["id"], result["category"],
                        result["english"], result["hindi"],
                        result["model"],
                        result["response_en"], result["response_hi"],
                        result["tokens_en"], result["tokens_hi"],
                        result["timestamp"]
                    ])
                    f.flush()

def process_models(questions: List[Dict]):
    if not check_system_resources():
        logging.warning("Continuing despite low memory, but performance may be impacted.")

    try:
        response = ollama.list()
        logging.info(f"Ollama API response: {response}")
        model_key = 'model' if 'model' in response.get('models', [{}])[0] else 'name'
        available_models = [model[model_key] for model in response.get('models', [])]
    except Exception as e:
        logging.error(f"Failed to list models: {str(e)}. Ensure Ollama server is running.")
        sys.exit(1)

    # Sort models: smallest to biggest
    size_priority = {
        "phi3:latest": 1,
        "gemma:7b": 2,
        "llama3:latest": 3,
        "deepseek-llm:latest": 4
    }
    sorted_models = sorted(MODELS, key=lambda m: size_priority.get(m, 100))
    valid_models = [model for model in sorted_models if model in available_models]

    if not valid_models:
        logging.error(f"No valid models found. Available: {available_models}")
        sys.exit(1)

    os.makedirs("results", exist_ok=True)

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID", "Category", "Question (EN)", "Question (HI)", "Model",
            "Response (EN)", "Response (HI)", "Tokens EN", "Tokens HI",
            "Timestamp"
        ])

    for model in valid_models:
        logging.info(f"Starting warm-up and processing for model: {model}")
        if not warm_up_model(model):
            logging.error(f"Skipping {model} due to warm-up failure.")
            continue
        process_model_questions(model, questions)

if __name__ == "__main__":
    logging.info("="*60)
    logging.info(f"LLM Evaluation - Models: {', '.join(MODELS)}")
    logging.info("="*60)

    if not os.path.exists(JSON_FILE):
        logging.error(f"JSON file {JSON_FILE} not found.")
        sys.exit(1)

    questions = load_questions(JSON_FILE)

    if len(questions) != 51:
        logging.warning(f"Expected 51 questions, found {len(questions)}")

    start_time = time.time()
    try:
        process_models(questions)
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        sys.exit(1)
    elapsed = time.time() - start_time

    logging.info(f"Evaluation completed in {elapsed/60:.2f} minutes")
    logging.info(f"Results saved to {OUTPUT_CSV}")
    print(f"Results saved to {OUTPUT_CSV}")
