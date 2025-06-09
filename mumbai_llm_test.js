import ollama from 'ollama';
import fs from 'fs/promises';
import { setTimeout } from 'timers/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import cliProgress from 'cli-progress';

// Setup __dirname in ES module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const QUESTIONS_PATH = path.join(__dirname, 'questions.json');
const RESULTS_DIR = path.join(__dirname, 'mumbai_llm_results');

// Configuration
const MODELS = ['llama3', 'mistral', 'phi3', 'gemma:7b', 'neural-chat'];
const QUESTIONS = JSON.parse(await fs.readFile(QUESTIONS_PATH, 'utf-8'));

// Create results directory if needed
try { await fs.mkdir(RESULTS_DIR); } catch {}

// Quality scoring
function scoreAnswer(answer, questionId) {
  const length = answer.length;
  let score = 50;
  if (length < 50) score -= 20;
  if (length > 500) score += 10;
  if (answer.includes("sorry") || answer.includes("not sure")) score -= 15;
  if (answer.split(' ').length > 100) score += 5;
  if (questionId === 1 && answer.toLowerCase().includes("vada pav")) score += 20;
  if (questionId === 12 && answer.toLowerCase().includes("financial capital")) score += 15;
  return Math.max(0, Math.min(100, score));
}

// Check if a model is available
async function isModelAvailable(model) {
  try {
    const response = await ollama.list();
    return response.models.some(m => m.name.startsWith(model));
  } catch {
    return false;
  }
}

// Main runner
async function runTests() {
  const finalReport = {
    start_time: new Date().toISOString(),
    models: {}
  };

  for (const model of MODELS) {
    console.log(`\n🚀 Testing ${model.toUpperCase()}...`);

    if (!(await isModelAvailable(model))) {
      console.warn(`⚠️ Model "${model}" not found in Ollama. Skipping.`);
      continue;
    }

    const bar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
    bar.start(QUESTIONS.length, 0);

    const modelResults = [];
    let totalScore = 0;

    for (const [index, q] of QUESTIONS.entries()) {
      try {
        // English
        const startEN = Date.now();
        const resEN = await ollama.chat({ model, messages: [{ role: 'user', content: q.english }] });
        const timeEN = (Date.now() - startEN) / 1000;
        const scoreEN = scoreAnswer(resEN.message.content, q.id);

        await setTimeout(1000);

        // Hindi
        const startHI = Date.now();
        const resHI = await ollama.chat({ model, messages: [{ role: 'user', content: q.hindi }] });
        const timeHI = (Date.now() - startHI) / 1000;
        const scoreHI = scoreAnswer(resHI.message.content, q.id);

        modelResults.push({
          id: q.id,
          category: q.category,
          english: {
            question: q.english,
            answer: resEN.message.content,
            time_sec: timeEN,
            chars: resEN.message.content.length,
            score: scoreEN
          },
          hindi: {
            question: q.hindi,
            answer: resHI.message.content,
            time_sec: timeHI,
            chars: resHI.message.content.length,
            score: scoreHI
          }
        });

        totalScore += (scoreEN + scoreHI) / 2;
      } catch (error) {
        modelResults.push({ id: q.id, error: error.message });
        console.error(`❌ Q${q.id} failed: ${error.message}`);
      }

      bar.update(index + 1);
      await setTimeout(1500); // pacing
    }

    bar.stop();

    const resultPath = path.join(RESULTS_DIR, `${model}_results.json`);
    await fs.writeFile(resultPath, JSON.stringify(modelResults, null, 2));

    finalReport.models[model] = {
      avg_score: (totalScore / QUESTIONS.length).toFixed(1),
      total_questions: QUESTIONS.length,
      details_file: `${model}_results.json`
    };
  }

  finalReport.end_time = new Date().toISOString();
  await fs.writeFile(
    path.join(RESULTS_DIR, 'final_report.json'),
    JSON.stringify(finalReport, null, 2)
  );

  console.log(`
🎉 All evaluations complete!
📁 Results saved in: ${RESULTS_DIR}
`);
}

runTests().catch(err => {
  console.error('💥 Unexpected error:', err);
});
