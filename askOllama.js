import { chat } from '@ollama/ai'; // Updated import
import fs from 'fs/promises';
import path from 'path';

// 1. Configuration
const RESULTS_FILE = './results.json';
const MODEL_NAME = 'llama3'; // Change as needed
const DELAY_BETWEEN_QUESTIONS = 2000; // 2 second delay

// 2. Initialize results file
async function initResultsFile() {
  try {
    await fs.access(RESULTS_FILE);
  } catch {
    await fs.writeFile(RESULTS_FILE, '[]');
  }
}

// 3. Verify Ollama connection
async function checkOllama() {
  try {
    const response = await chat({ 
      model: MODEL_NAME,
      messages: [{ role: 'user', content: 'ping' }],
      options: { timeout: 5000 }
    });
    return true;
  } catch (error) {
    console.error('❌ Ollama connection failed:', error.message);
    process.exit(1);
  }
}

// 4. Main processing
async function processQuestions() {
  await initResultsFile();
  await checkOllama();

  const questions = JSON.parse(await fs.readFile('./questions.json'));
  const existingResults = JSON.parse(await fs.readFile(RESULTS_FILE));
  
  for (const [index, questionObj] of questions.entries()) {
    try {
      console.log(`\n❓ Question ${index + 1}/${questions.length}: ${questionObj.english}`);
      
      const response = await chat({
        model: MODEL_NAME,
        messages: [{ role: 'user', content: questionObj.english }],
        options: { timeout: 30000 } // 30 second timeout
      });

      const result = {
        id: questionObj.id,
        question: questionObj.english,
        answer: response.message.content,
        timestamp: new Date().toISOString()
      };

      console.log(`💡 Answer (${response.message.content.length} chars): ${response.message.content.substring(0, 100)}...`);

      existingResults.push(result);
      await fs.writeFile(RESULTS_FILE, JSON.stringify(existingResults, null, 2));
      
      await new Promise(resolve => setTimeout(resolve, DELAY_BETWEEN_QUESTIONS));
    } catch (error) {
      console.error(`⚠️ Failed on question ${index + 1}:`, error.message);
    }
  }
}

processQuestions().catch(console.error);