#!/usr/bin/env python3
"""
Autonomer Multi-Agent-Forschungszyklus mit dynamischem Modell-Swapping
über die OpenAI-kompatible Schnittstelle von llama.cpp.
LOKAL ANGEPASST FÜR DYNAMISCHE PFADE UND ROUTER-MODELL-NAMEN.
"""

import json
import os
import subprocess
import tempfile
import re
import time
import threading
import psutil
import hashlib
import shlex
import ast
import sys
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

# ==============================================================================
# NEU: OpenAI-Client statt ollama
# ==============================================================================
from openai import OpenAI

# ==============================================================================
# KONFIGURATION
# ==============================================================================
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8080/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")

# WICHTIG: Diese Namen müssen mit den Dateinamen im --models-dir des Routers übereinstimmen (ohne .gguf)
MODEL_ARCHITECT   = "nemotron"          # Dein Architekt (entspricht nemotron.gguf)
MODEL_PROGRAMMER  = "vibethinker"       # Dein Programmierer (entspricht vibethinker.gguf)
MODEL_AUDITOR     = "nemotron"          # Dein Auditor nemotron-cascade-2

# Weitere Konfigurationen
CHECKPOINT_FILE = "experiment_checkpoint.json"
RESULTS_DIR = Path("results")
MAX_RETRIES = 3
MAX_SUBDIVISION_DEPTH = 3
MAX_SUBDIVISIONS_PER_TASK = 8
OLLAMA_TIMEOUT = 3000
CONCURRENT_REQUESTS = 1

RESULTS_DIR.mkdir(exist_ok=True)

llm_semaphore = threading.Semaphore(CONCURRENT_REQUESTS)

# ==============================================================================
# NEU: OpenAI-Client initialisieren
# ==============================================================================
openai_client = OpenAI(base_url=LLM_API_BASE, api_key=OPENAI_API_KEY)

# ==============================================================================
# UTILITIES: Robust Logic & Atomic I/O
# ==============================================================================

def query_model(model, prompt, system="", timeout=OLLAMA_TIMEOUT):
    print(f"[DEBUG] Querying model {model} via OpenAI API...")
    for attempt in range(MAX_RETRIES):
        acquired = llm_semaphore.acquire(timeout=120)
        if not acquired:
            print(f"[Model] Semaphore timeout on {model} (Attempt {attempt+1})")
            continue

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    openai_client.chat.completions.create,
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=timeout
                )
                res = future.result(timeout=timeout + 5)
                return res.choices[0].message.content

        except TimeoutError:
            print(f"[Model] Timeout on {model} (Attempt {attempt+1})")
            future.cancel()
        except Exception as e:
            print(f"[Model] Error: {e} (Attempt {attempt+1})")
        finally:
            llm_semaphore.release()
        time.sleep(2 ** attempt)

    return "ERROR_PERSISTENT"


def robust_json_parse(text):
    if not text or "ERROR_PERSISTENT" in text:
        if text: text = text.split("<arg_key>output}<arg_value>")[-1]
    elif "</thought>" in text:
        text = text.split("</thought>")[-1]
    else:
        text = re.sub(r"aleigh.*?</tool_call>", "", text, flags=re.DOTALL)
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)

    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text.strip())

    try:
        start = text.rfind('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            json_str = text[start:end+1]
            try: return json.loads(json_str)
            except json.JSONDecodeError: pass

        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            start = text.find('[')
            end = text.rfind(']')
        if start == -1 or end == -1: return None
        
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(json_str)
                if isinstance(parsed, (dict, list)): return parsed
            except (ValueError, SyntaxError): pass
        return None
    except Exception:
        return None


def extract_python_code(text):
    if not text: return ""

    text = re.sub(r"aleigh.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)

    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()

    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith(("import ", "from ", "def ", "class ", "print(", "#")):
            print(f"[INFO] Found code-like start at line {i}. Using remaining text as code.")
            return '\n'.join(lines[i:]).strip()

    print(f"[WARN] No python code fence or heuristic match found.")
    return ""


def atomic_write_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as tf:
        json.dump(data, tf, indent=2)
        temp_name = tf.name
    os.replace(temp_name, path)


# ==============================================================================
# ENVIRONMENT & SYSTEM MANAGEMENT (LOKAL ANGEPASST)
# ==============================================================================

class EnvironmentAuditor:
    # Gudhi und Jax sind optional, da sie schwer zu installieren sind
    REQUIRED_LIBS = ["numpy", "scipy", "matplotlib", "sympy", "networkx", "psutil"]
    OPTIONAL_LIBS = ["gudhi", "ripser", "jax", "jax_md"]

    @staticmethod
    def run_readiness_report():
        print("--- RUNNING PRE-FLIGHT ENVIRONMENT AUDIT ---")
        missing = []
        for lib in EnvironmentAuditor.REQUIRED_LIBS:
            try: __import__(lib); print(f"[Ready] {lib}")
            except ImportError: missing.append(lib); print(f"[MISSING] {lib} (Required!)")
            
        for lib in EnvironmentAuditor.OPTIONAL_LIBS:
            try: __import__(lib); print(f"[Ready] {lib} (Optional)")
            except ImportError: print(f"[MISSING] {lib} (Optional - Skript läuft trotzdem)")
            
        has_bwrap = shutil.which("bwrap") is not None
        if has_bwrap:
            print("[Ready] bubblewrap (bwrap) gefunden")
        else:
            print("[WARN] bubblewrap (bwrap) nicht gefunden. Code wird ohne Sandbox ausgeführt!")

        if missing:
            print(f"--- FEHLER: Benötigte Bibliotheken fehlen: {missing} ---")
            return False
        print("--- ENVIRONMENT CERTIFIED: READY ---")
        return True


class SystemManager:
    K_MAX = MAX_SUBDIVISIONS_PER_TASK
    D_MAX = MAX_SUBDIVISION_DEPTH
    N_STAR_COEFFICIENT = K_MAX ** D_MAX

    @staticmethod
    def pre_flight_check(n_est_kb, confidence_rho):
        Sv_kb = psutil.virtual_memory().available // 1024
        sigma_buffer = 0.15
        kappa = 1.0 + 0.5 * (1.0 - confidence_rho)
        L_eff_kb = (Sv_kb / kappa) * (1.0 - sigma_buffer)
        delta = n_est_kb - L_eff_kb

        n_star = SystemManager.N_STAR_COEFFICIENT * L_eff_kb if L_eff_kb > 0 else float('inf')
        if n_est_kb > n_star: return "reject", L_eff_kb
        if confidence_rho < 0.35: return "reject", L_eff_kb
        if delta <= 0 and confidence_rho >= 0.65: return "execute", L_eff_kb
        if delta > 0.20 * L_eff_kb: return "reject", L_eff_kb
        return "subdivide", L_eff_kb

    @staticmethod
    def calculate_optimal_k(N_kb, L_eff_kb):
        k_naive = max(2, int(N_kb / (L_eff_kb * 0.8)) + 1)
        return min(k_naive, MAX_SUBDIVISIONS_PER_TASK)


# ==============================================================================
# KNOWLEDGE REPORTER
# ==============================================================================

class KnowledgeReporter:
    def __init__(self, storage_path="master_knowledge.json"):
        self.path = Path(storage_path)
        self.memory = self._load()
        self._lock = threading.Lock()

    def _load(self):
        if self.path.exists():
            try: return json.loads(self.path.read_text())
            except Exception: return {}
        return {}

    def update_manifold(self, cycle_data, semantic_val):
        cid = cycle_data['id']
        summary = query_model(MODEL_AUDITOR,
                              f"Summarize Cycle {cid}. Result: {semantic_val.get('reason', 'N/A')}.",
                              "Registrar.")
        if "ERROR_PERSISTENT" in summary: summary = "Summary generation failed."
        with self._lock:
            invariants = semantic_val.get("invariants", ["Unknown-Stability"])
            self.memory[cid] = {"invariants": invariants, "summary": summary}
            atomic_write_json(self.path, self.memory)

    def get_context(self):
        with self._lock:
            recent_ids = list(self.memory.keys())[-10:]
            context = "=== RECENT INVARIANTS ===\n"
            for cid in recent_ids:
                data = self.memory[cid]
                context += f"Cycle {cid}: {', '.join(data['invariants'])}\n"
                context += f"Summary {cid}: {data['summary']}\n"
            return context


# ==============================================================================
# PIPELINE (LOKAL ANGEPASST)
# ==============================================================================

class Auditor:
    @staticmethod
    def verify_pullback(history, proof):
        prompt = (f"ORIGINAL: {proof}\nRETRIEVED: {history}\n"
                  f"Compare logically. Return JSON: {{\"faithful\": bool, \"rho\": float}}")
        raw_output = query_model(MODEL_AUDITOR, prompt, "Security Auditor.")
        parsed = robust_json_parse(raw_output)
        if not parsed:
            print(f"\n[DIAGNOSTICS] JSON parsing failed. Raw LLM Output:\n{raw_output}\n")
            return {"faithful": False, "rho": 0.0}
        return parsed

    @staticmethod
    def semantic_validation(hypothesis, stdout):
        prompt = (f"HYPOTHESIS: {hypothesis}\nOUTPUT: {stdout[-4000:]}\n"
                  f"Does the output support the hypothesis? "
                  f"Return JSON: {{\"hypothesis_holds\": bool, \"reason\": \"str\", \"invariants\": [\"str\"]}}")
        return robust_json_parse(query_model(MODEL_AUDITOR, prompt, "Validator.")) or \
               {"hypothesis_holds": False, "reason": "Parse error", "invariants": []}


def run_autonomous_cycle(cycle_data, reporter, depth=0, feedback=""):
    if depth > MAX_SUBDIVISION_DEPTH:
        return {"id": cycle_data['id'], "status": "recursion_limit"}

    cid = cycle_data['id']
    print(f"\n>>> [Cycle {cid}] Depth={depth}")
    if feedback: print(f"[DEBUG] Retrying with feedback: {feedback[:100]}...")
    
    history = reporter.get_context()

    if len(history.strip().split('\n')) <= 1:
        pullback = {"faithful": True, "rho": 0.8}
    else:
        print(f"[DEBUG] Verifying pullback for Cycle {cid}...")
        pullback = Auditor.verify_pullback(history, cycle_data['proof'])
    
    if not pullback.get('faithful'):
        print(f"[ERROR] Integrity failed for Cycle {cid}.")
        return {"id": cid, "status": "integrity_failed"}

    print(f"[DEBUG] Generating blueprint for Cycle {cid}...")
    blueprint_prompt = (
        f"HISTORY:\n{history}\n\n"
        f"TASK / HYPOTHESIS:\n{cycle_data['hypothesis']}\n\n"
        f"PREVIOUS FAILURE FEEDBACK (if any):\n{feedback}\n\n"
        f"You are designing a computational experiment. Provide a DETAILED blueprint containing:\n"
        f"1. ALGORITHM: Step-by-step algorithm design and pseudocode\n"
        f"2. DATA STRUCTURES: Key data structures and their relationships\n"
        f"3. LIBRARIES: Required Python libraries (prefer numpy, scipy, networkx)\n"
        f"4. METHOD: How the hypothesis will be computationally tested\n"
        f"5. OUTPUT: Exact print statements the script should produce\n\n"
        f"After your full blueprint, you MUST include this exact footer line "
        f"(required by our orchestration system):\n"
        f"ESTIMATED_MEMORY_KIB: <integer>\n"
        f"where <integer> is your estimate of peak memory usage in KiB."
    )
    blueprint = query_model(MODEL_ARCHITECT, blueprint_prompt, "Senior Architect.")
    if "ERROR_PERSISTENT" in blueprint:
        return {"id": cid, "status": "architect_timeout"}

    blueprint_lines = [l for l in blueprint.strip().split('\n') if l.strip() and not l.strip().startswith("ESTIMATED_MEMORY_KIB")]
    if len(blueprint_lines) < 3:
        return {"id": cid, "status": "blueprint_too_sparse"}

    n_est_kb = 10240
    n_match = re.search(r"ESTIMATED_MEMORY_KIB[:\s]*(\d+)", blueprint)
    if n_match: n_est_kb = max(512, min(2097152, int(n_match.group(1))))

    decision, L_eff_kb = SystemManager.pre_flight_check(n_est_kb, pullback.get('rho', 0.5))

    if decision == "subdivide":
        k_star = SystemManager.calculate_optimal_k(n_est_kb, L_eff_kb)
        decompose_prompt = (
            f"Decompose this hypothesis into exactly {k_star} independent sub-hypotheses. "
            f"Output STRICTLY valid JSON: {{\"sub_hypotheses\": [\"sub1\", \"sub2\", ...]}}\n"
            f"HYPOTHESIS: {cycle_data['hypothesis']}"
        )
        decompose_result = query_model(MODEL_ARCHITECT, decompose_prompt, "Task Decomposition Specialist.")
        if "ERROR_PERSISTENT" in decompose_result: return {"id": cid, "status": "decomposition_timeout"}
        
        sub_hypotheses_data = robust_json_parse(decompose_result)
        if not sub_hypotheses_data or "sub_hypotheses" not in sub_hypotheses_data:
            return {"id": cid, "status": "decompose_failed"}

        sub_results = []
        for i, sub_hyp in enumerate(sub_hypotheses_data["sub_hypotheses"][:k_star]):
            sub_data = {**cycle_data, "id": f"{cid}_sub{i}", "hypothesis": sub_hyp}
            sub_results.append(run_autonomous_cycle(sub_data, reporter, depth=depth + 1))

        n_success = sum(1 for r in sub_results if r and r.get("status") == "success")
        return {"id": cid, "status": "subdivided_complete", "sub_results": sub_results, "success_rate": n_success / len(sub_results) if sub_results else 0}

    if decision == "reject":
        return {"id": cid, "status": "rejected"}

    # Programmierer generiert Code
    print(f"[DEBUG] Generating Python code for Cycle {cid}...")
    programmer_prompt = (
        f"You are a Computational Physicist. Write a COMPLETE, SELF-CONTAINED Python script.\n\n"
        f"HYPOTHESIS TO TEST:\n{cycle_data['hypothesis']}\n\n"
        f"IMPLEMENTATION BLUEPRINT:\n{blueprint}\n\n"
        f"PREVIOUS FAILURE FEEDBACK (if any):\n{feedback}\n\n"
        f"MANDATORY REQUIREMENTS:\n"
        f"- Output the code inside a ```python code block\n"
        f"- Import ALL needed libraries at the top\n"
        f"- DO NOT USE PLACEHOLDERS. The mathematical computation MUST be actually implemented.\n"
        f"- End by printing exactly one of:\n"
        f"  HYPOTHESIS_CONFIRMED: True -- <reason>\n"
        f"  HYPOTHESIS_CONFIRMED: False -- <reason>\n"
        f"- Use only libraries available in standard scientific Python (numpy, sympy, scipy, networkx)."
    )
    code_raw = query_model(MODEL_PROGRAMMER, programmer_prompt, "Computational Physicist.")
    if "ERROR_PERSISTENT" in code_raw: return {"id": cid, "status": "programmer_timeout"}

    code = extract_python_code(code_raw)

    if not code:
        result_dir = RESULTS_DIR / cid
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "failed_output_raw.txt").write_text(code_raw)
        return {"id": cid, "status": "no_code_extracted"}

    if not any(kw in code for kw in ["import ", "from ", "def ", "class "]):
        return {"id": cid, "status": "invalid_code_structure"}

    result_dir = RESULTS_DIR / cid
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "blueprint.md").write_text(blueprint)
    (result_dir / "code.py").write_text(code)

    # ======================================================================
    # LOKALE AUSFÜHRUNG (Dynamisch angepasst)
    # ======================================================================
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "exp.py"
        script_path.write_text(code)

        # Finde den aktuellen Python-Interpreter dynamisch
        python_exec = sys.executable
        
        # Prüfe ob bwrap existiert
        use_sandbox = shutil.which("bwrap") is not None

        if use_sandbox:
            # Dynamische Ermittlung des VENV-Pfads basierend auf dem Python-Interpreter
            venv_path = os.path.dirname(os.path.dirname(python_exec))
            
            bwrap_cmd = [
                "bwrap",
                "--ro-bind", "/usr", "/usr",
                "--ro-bind", "/bin", "/bin",
                "--ro-bind", "/lib", "/lib",
                "--ro-bind", "/lib64", "/lib64",
                "--ro-bind", "/etc", "/etc",
                "--ro-bind", venv_path, venv_path,  # Dynamischer VENV!
                "--proc", "/proc",
                "--dev-bind", "/dev/null", "/dev/null",
                "--dev-bind", "/dev/urandom", "/dev/urandom",
                "--dev-bind", "/dev/random", "/dev/random",
                "--tmpfs", "/tmp",
                "--bind", tmpdir, "/app",
                "--unshare-all",
                "--new-session",
                "--die-with-parent",
                "--cap-drop", "ALL",
                python_exec, "/app/exp.py"
            ]
            ulimit_prefix = f"ulimit -v {int(L_eff_kb)} -m {int(L_eff_kb)} -u 2048 -t 60 && "
            cmd = ["bash", "-c", ulimit_prefix + shlex.join(bwrap_cmd)]
        else:
            # Fallback: Direkte Ausführung ohne Sandbox, falls bwrap fehlt
            print("[WARN] Führe Code OHNE bubblewrap-Sandbox aus!")
            cmd = [python_exec, str(script_path)]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            stdout = res.stdout + res.stderr
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or "") + (e.stderr or "") + "\n[TIMEOUT_EXPIRED: process exceeded 120s]"
        except Exception as e:
            stdout = f"\n[EXECUTION_ERROR: {str(e)}]"

    (result_dir / "stdout.txt").write_text(stdout[:50000])

    valid = Auditor.semantic_validation(cycle_data['hypothesis'], stdout)
    
    if valid.get("hypothesis_holds"):
        reporter.update_manifold(cycle_data, valid)
        return {"id": cid, "status": "success"}
    else:
        max_retries = 2
        if "retry_count" not in cycle_data: cycle_data["retry_count"] = 0
        if cycle_data["retry_count"] < max_retries:
            cycle_data["retry_count"] += 1
            new_feedback = f"Execution Output:\n{stdout[-2000:]}\n\nValidation Reason:\n{valid.get('reason')}"
            return run_autonomous_cycle(cycle_data, reporter, depth=depth, feedback=new_feedback)
        return {"id": cid, "status": "failed", "reason": valid.get("reason")}


def main():
    if not EnvironmentAuditor.run_readiness_report():
        print("\n[ERROR] Bitte installiere die fehlenden Bibliotheken: pip install numpy scipy matplotlib sympy networkx psutil openai")
        return

    log_file = next(
        (p / "sheaf_multiagent_research_log.md"
         for p in [Path("."), Path("erledit"), Path("test_code")]
         if (p / "sheaf_multiagent_research_log.md").exists()),
        None
    )
    
    # Fallback: Erstelle eine Test-Log-Datei, falls keine existiert
    if not log_file:
        print("[WARN] Keine Log-Datei gefunden. Erstelle eine Dummy-Datei für einen Testlauf...")
        log_file = Path("sheaf_multiagent_research_log.md")
        log_file.write_text("""---
### Cycle 1 - Test_Hypothesis
**Hypothesis:** The sum of angles in a flat triangle equals 180 degrees.
**Proof:** Basic Euclidean geometry.
""")

    content = log_file.read_text(encoding="utf-8")
    raw_cycles = [c for c in re.split(r'^---\s*$', content, flags=re.MULTILINE)
                  if "**Hypothesis:**" in c]

    reporter, cp_path = KnowledgeReporter(), Path(CHECKPOINT_FILE)
    done_ids = json.loads(cp_path.read_text()).get("done", []) if cp_path.exists() else []

    pending_tasks = []
    for raw in raw_cycles:
        id_match = re.search(r"### Cycle (\d+)\s*-\s*(.*)", raw)
        if id_match:
            cycle_num = id_match.group(1)
            title_slug = re.sub(r"[^a-zA-Z0-9]+", "_", id_match.group(2).strip())[:30].strip("_")
            cid = f"{cycle_num}_{title_slug}"
        else:
            cid = hashlib.sha256(raw.encode()).hexdigest()[:16]

        if cid in done_ids: continue

        m_hyp = re.search(r"\*\*hypothesis[:\s]*\*\*[:\s]*(.*?)(?=\n\*\*)", raw, re.DOTALL | re.IGNORECASE)
        m_proof = re.search(r"\*\*proof[:\s]*\*\*[:\s]*(.*?)(?=\nverdict|\n\*\*|---|$)", raw, re.DOTALL | re.IGNORECASE)
        if not m_hyp: continue

        pending_tasks.append({
            "id": cid,
            "hypothesis": m_hyp.group(1).strip(),
            "proof": m_proof.group(1).strip() if m_proof else ""
        })

    print(f"Batch Start. {len(pending_tasks)} pending.")

    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool:
        future_to_cid = {
            pool.submit(run_autonomous_cycle, data, reporter): data["id"]
            for data in pending_tasks
        }
        for future in as_completed(future_to_cid):
            cid = future_to_cid[future]
            try:
                res = future.result()
                if res and res.get("status") in ["success", "subdivided_complete"]:
                    done_ids.append(cid)
                    atomic_write_json(cp_path, {"done": done_ids})
                    print(f"Cycle {cid} OK.")
                else:
                    print(f"Cycle {cid} finished with status: {res.get('status', 'unknown')}")
            except Exception as e:
                print(f"Cycle {cid} raised a fatal exception: {e}")


if __name__ == "__main__":
    main()
