@echo off
echo === Mumbai LLM Evaluation Setup ===
echo This will install dependencies and prepare Ollama models
echo Project Configuration:
echo - 51 Questions (English/Hindi)
echo - 5 Tested LLMs
echo.

:: Verify Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

:: Install Python packages
echo Step 1: Installing Python dependencies...
pip install "tqdm>=4.66.0" && echo [OK] tqdm installed || echo [FAIL] tqdm installation failed
pip install ollama && echo [OK] ollama installed || echo [FAIL] ollama installation failed

:: Install Ollama models (lightest to heaviest)
echo.
echo Step 2: Installing Ollama models...
echo Recommended models for Mumbai queries:
set models=phi3 falcon:7b gemma:7b deepseek-llm llama3

for %%m in (%models%) do (
    echo Pulling: %%m
    ollama pull %%m && echo [OK] %%m installed || echo [FAIL] Failed to install %%m
    timeout /t 3 >nul  :: Delay between pulls
)

:: Verify questions.json
echo.
echo Step 3: Verifying question files...
if exist "questions.json" (
    for /f %%C in ('type "questions.json" ^| find /c /v ""') do (
        if %%C geq 51 (
            echo [OK] Found questions.json with 51+ lines
        ) else (
            echo [WARNING] questions.json appears incomplete (only %%C lines)
        )
    )
) else (
    echo [ERROR] Missing questions.json - please add it to the folder
)

:: Create results directory
mkdir results 2>nul && echo [OK] Created results directory || echo [WARNING] Results directory exists

echo.
echo === Setup Summary ===
echo Python Packages:
pip list | findstr "tqdm ollama"
echo.
echo Installed Models:
ollama list
echo.
echo Question File: %cd%\questions.json
echo Output Directory: %cd%\results
echo.

set /p run_test=Would you like to run the Mumbai LLM evaluation now? (Y/N): 
if /i "%run_test%"=="Y" (
    echo Starting evaluation...
    python mumbai_llm_test.py
) else (
    echo You can manually run later with: python mumbai_llm_test.py
)

pause
