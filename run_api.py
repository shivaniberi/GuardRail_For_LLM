import sys
import os
import subprocess
 
# ── Free port 8000 before binding ─────────────────────────────────────────────
def free_port(port: int):
    result = subprocess.run(
        f'netstat -ano | findstr :{port}',
        shell=True, capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if 'LISTENING' in line:
            pid = line.strip().split()[-1]
            subprocess.run(f'taskkill /PID {pid} /F', shell=True)
            print(f"Killed existing process {pid} on port {port}")
 
free_port(8000)
 
# ── Add core/ to Python path so relative imports work ─────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
 
# Patch the relative imports in core files
from core import guardrail_implementation
from core import ml_input_guardrail
from core import ollama_client
 
import uvicorn
 
if __name__ == "__main__":
    uvicorn.run("core.api:app", host="0.0.0.0", port=8000, reload=False)