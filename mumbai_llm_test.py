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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mumbai_llm.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_questions(json_file: str) -> List[Dict]:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Extract and flatten all questions from categories
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

MODEL = "falcon:7b-instruct"

def warm_up_model(model: str, max_retries: int = 3) -> bool:
    warmup_question = "What is Mumbai famous for? Answer in one sentence."
    for attempt in range(max_retries):
        try:
            res = ollama.generate(
                model=model,
                prompt=warmup_question,
                options={"num_predict": 20}
            )
            if res.get("response"):
                logging.info(f"{model} warm-up successful. Response: {res['response'][:50]}...")
                return True
        except Exception as e:
            logging.warning(f"{model} warm-up failed ({attempt+1}/{max_retries}): {e}")
            time.sleep(3)
    return False

def query_ollama(model: str, prompt: str, retries: int = 3) -> Optional[str]:
    options = {
        "num_ctx": 2048,
        "temperature": 0.3,
        "seed": 42,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_predict": 512
    }

    for attempt in range(retries + 1):
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
            logging.warning(f"{model} query failed (attempt {attempt+1}): {e}")
            time.sleep(5 * (attempt + 1))
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

def process_model(model: str, questions: List[Dict]):
    if not warm_up_model(model):
        logging.error(f"{model} could not be loaded. Exiting.")
        sys.exit(1)

    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/falcon_7b_instruct_results_{timestamp}.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID", "Category", "Question (EN)", "Question (HI)", "Model",
            "Response (EN)", "Response (HI)", "Tokens EN", "Tokens HI",
            "Timestamp"
        ])
        
        results = []
        lock = Lock()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_question, model, q): q for q in questions}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {model}"):
                result = future.result()
                results.append(result)
                
                with lock:
                    writer.writerow([
                        result["id"], result["category"],
                        result["english"], result["hindi"],
                        result["model"],
                        result["response_en"], result["response_hi"],
                        result["tokens_en"], result["tokens_hi"],
                        result["timestamp"]
                    ])
                    f.flush()

if __name__ == "__main__":
    logging.info("="*60)
    logging.info(f"Falcon 7B Instruct Evaluation - Model: {MODEL}")
    logging.info("="*60)

    questions = load_questions("questions.json")

    if len(questions) != 51:
        logging.warning(f"Expected 51 questions, found {len(questions)}")

    start_time = time.time()
    process_model(MODEL, questions)
    elapsed = time.time() - start_time
    
    logging.info(f"Evaluation completed in {elapsed/60:.2f} minutes")
    logging.info("Results saved with timestamp in filename")