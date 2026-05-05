#!/usr/bin/env python3
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
import concurrent.futures
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from ollama import Client

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_ARCHITECT = "nemotron-cascade-2:latest"
MODEL_PROGRAMMER = "qwen3.6:latest"
MODEL_AUDITOR = "nemotron-cascade-2:latest"

CHECKPOINT_FILE = "experiment_checkpoint.json"
RESULTS_DIR = Path("results")

MAX_RETRIES = 3
MAX_SUBDIVISION_DEPTH = 3
MAX_SUBDIVISIONS_PER_TASK = 8
OLLAMA_TIMEOUT = 300
CONCURRENT_REQUESTS = 1

client = Client(host=OLLAMA_HOST)
RESULTS_DIR.mkdir(exist_ok=True)

# Controlled concurrency
ollama_semaphore = threading.Semaphore(CONCURRENT_REQUESTS)

# ==============================================================================
# UTILITIES: Robust Logic & Atomic I/O
# ==============================================================================
def query_model(model, prompt, system="", timeout=OLLAMA_TIMEOUT):
    """Thread-safe query with true timeout enforcement and guaranteed semaphore release."""
    for attempt in range(MAX_RETRIES):
        # Acquire semaphore outside the thread to prevent thread leakage on timeout
        acquired = ollama_semaphore.acquire(timeout=timeout + 30)
        if not acquired:
            print(f"[Model] Semaphore timeout on {model} (Attempt {attempt+1})")
            continue
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(client.generate, model=model, prompt=prompt, system=system, stream=False)
                res = future.result(timeout=timeout)
                return res.get('response', '') if isinstance(res, dict) else getattr(res, 'response', '')
        except TimeoutError:
            print(f"[Model] Timeout on {model} (Attempt {attempt+1})")
            # Attempt to cancel the hanging future
            future.cancel()
        except Exception as e:
            print(f"[Model] Error: {e} (Attempt {attempt+1})")
        finally:
            # ALWAYS release the semaphore
            ollama_semaphore.release()
            
        time.sleep(2 ** attempt)
        
    return "ERROR_PERSISTENT"

def robust_json_parse(text):
    """Surgically extracts JSON. Uses ast.literal_eval as a fallback for single-quoted dicts."""
    if not text or "ERROR_PERSISTENT" in text:
        return None
    
    # NEW: Aggressively strip reasoning blocks that might contain distracting JSON-like snippets
    # Some models don't start with <think> but end with </think>
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "</thought>" in text:
        text = text.split("</thought>")[-1]
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    
    # Strip markdown code blocks if present around the JSON
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text.strip())
    
    # Find all potential JSON objects and try to parse them from the end (usually the final answer)
    try:
        # Simple heuristic: try the last { ... } block first
        start = text.rfind('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            json_str = text[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Fallback to the first { ... } block if the last one failed
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            # Try parsing arrays
            start = text.find('[')
            end = text.rfind(']')
            if start == -1 or end == -1:
                return None
                
        json_str = text[start:end+1]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: Python's ast handles single quotes and unquoted keys gracefully
            try:
                parsed = ast.literal_eval(json_str)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except (ValueError, SyntaxError):
                pass
            
            return None
    except Exception:
        return None

def extract_python_code(text):
    """Extracts the first ```python ... ``` block, or falls back to raw text."""
    if not text:
        return ""
    
    # Strip reasoning tags if they leaked here
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    
    # Try finding markdown blocks
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Heuristic: Find first line starting with import or def
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith(("import ", "from ", "def ", "class ")):
            print(f"[INFO] Found code-like start at line {i}. Using remaining text as code.")
            return '\n'.join(lines[i:]).strip()

    # Fallback if model forgot code fences
    print(f"[WARN] No python code fence or heuristic match found in model output. Raw text length: {len(text)}")
    if len(text) < 500:
        print(f"[DEBUG] Raw text: {text[:200]}...")
    return text.strip()

def atomic_write_json(path, data):
    """Prevents checkpoint corruption during crashes."""
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as tf:
        json.dump(data, tf, indent=2)
        temp_name = tf.name
    os.replace(temp_name, path)

# ==============================================================================
# ENVIRONMENT & SYSTEM MANAGEMENT
# ==============================================================================
class EnvironmentAuditor:
    REQUIRED_LIBS = ["sympy", "gudhi", "ripser", "jax", "jax_md", "networkx", "psutil", "ollama"]

    @staticmethod
    def run_readiness_report():
        print("--- RUNNING PRE-FLIGHT ENVIRONMENT AUDIT ---")
        missing = []
        for lib in EnvironmentAuditor.REQUIRED_LIBS:
            try:
                __import__(lib); print(f"[Ready] {lib}")
            except ImportError:
                missing.append(lib); print(f"[MISSING] {lib}")

        try:
            subprocess.run(["bwrap", "--version"], capture_output=True, check=True)
            print("[Ready] bubblewrap (bwrap)")
        except Exception:
            missing.append("bwrap"); print("[MISSING] bubblewrap")

        if missing:
            return False
        print("--- ENVIRONMENT CERTIFIED: READY ---")
        return True

class SystemManager:
    # Mathematische Konstanten aus dem Theorem
    K_MAX = MAX_SUBDIVISIONS_PER_TASK  # 8
    D_MAX = MAX_SUBDIVISION_DEPTH      # 3
    N_STAR_COEFFICIENT = K_MAX ** D_MAX # 8^3 = 512

    @staticmethod
    def pre_flight_check(n_est_kb, confidence_rho):
        Sv_kb = psutil.virtual_memory().available // 1024
        sigma_buffer = 0.15
        kappa = 1.0 + 0.5 * (1.0 - confidence_rho)
        L_eff_kb = (Sv_kb / kappa) * (1.0 - sigma_buffer)
        delta = n_est_kb - L_eff_kb

        # ==============================================================================
        # THEOREM 1: EARLY REJECT TO PREVENT DEATH SPIRAL (N > N*)
        # Wenn N > 512 / L_eff, wird selbst bei max. Verzweigung (8) und max. Tiefe (3) 
        # jede Teilaufgabe immer noch L_eff überschreiten. 
        # Subdivision wäre reine Verschwendung von Rechenzeit.
        # ==============================================================================
        n_star = SystemManager.N_STAR_COEFFICIENT * L_eff_kb if L_eff_kb > 0 else float('inf')
        if n_est_kb > n_star:
            return "reject", L_eff_kb

        # ==============================================================================
        # INVARIANT CHECK: Kein Overshoot erlaubt (wie vorher bewiesen)
        # ==============================================================================
        if confidence_rho < 0.35:
            return "reject", L_eff_kb
            
        if delta <= 0 and confidence_rho >= 0.65:
            return "execute", L_eff_kb

        if delta > 0.20 * L_eff_kb:
            return "reject", L_eff_kb

        # ==============================================================================
        # SAFE SUBDIVISION: Nur wenn Delta > 0 UND N <= N*
        # ==============================================================================
        return "subdivide", L_eff_kb

    @staticmethod
    def calculate_optimal_k(N_kb, L_eff_kb):
        """Sanity-capped subdivision factor."""
        k_naive = max(2, int(N_kb / (L_eff_kb * 0.8)) + 1)
        return min(k_naive, MAX_SUBDIVISIONS_PER_TASK)


# ==============================================================================
# KNOWLEDGE REPORTER (Thread-Safe, Non-Blocking Network Calls)
# ==============================================================================
class KnowledgeReporter:
    def __init__(self, storage_path="master_knowledge.json"):
        self.path = Path(storage_path)
        self.memory = self._load()
        self._lock = threading.Lock()

    def _load(self):
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                return {}
        return {}

    def update_manifold(self, cycle_data, semantic_val):
        cid = cycle_data['id']
        
        # FIX #10: Query model OUTSIDE the lock to prevent blocking other threads
        summary = query_model(MODEL_AUDITOR, f"Summarize Cycle {cid}. Result: {semantic_val.get('reason', 'N/A')}.", "Registrar.")
        
        if "ERROR_PERSISTENT" in summary:
            summary = "Summary generation failed."
            
        with self._lock:
            # FIX #15: Extract invariants dynamically from data instead of hardcoding
            invariants = semantic_val.get("invariants", ["Unknown-Stability"])
            self.memory[cid] = {
                "invariants": invariants,
                "summary": summary
            }
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
# PIPELINE: Real Subdivision & Hardened Sandbox
# ==============================================================================
class Auditor:
    @staticmethod
    def verify_pullback(history, proof):
        prompt = f"ORIGINAL: {proof}\nRETRIEVED: {history}\nCompare logically. Return JSON: {{\"faithful\": bool, \"rho\": float}}"
        
        raw_output = query_model(MODEL_AUDITOR, prompt, "Security Auditor.")
        parsed = robust_json_parse(raw_output)
        
        if not parsed:
            print(f"\n[DIAGNOSTICS] JSON parsing failed or empty. Raw LLM Output:\n{raw_output}\n")
            return {"faithful": False, "rho": 0.0}
            
        return parsed

    @staticmethod
    def semantic_validation(hypothesis, stdout):
        prompt = f"HYPOTHESIS: {hypothesis}\nOUTPUT: {stdout[-4000:]}\nDoes the output support the hypothesis? Return JSON: {{\"hypothesis_holds\": bool, \"reason\": \"str\", \"invariants\": [\"str\"]}}"
        return robust_json_parse(query_model(MODEL_AUDITOR, prompt, "Validator.")) or {"hypothesis_holds": False, "reason": "Parse error", "invariants": []}

def run_autonomous_cycle(cycle_data, reporter, depth=0):
    if depth > MAX_SUBDIVISION_DEPTH:
        return {"id": cycle_data['id'], "status": "recursion_limit"}
        
    cid = cycle_data['id']
    print(f"\n>>> [Cycle {cid}] Depth={depth}")
    
    history = reporter.get_context()
    
    # FIX: If history is empty (just the header), don't fail integrity check
    if len(history.strip().split('\n')) <= 1:
        pullback = {"faithful": True, "rho": 0.8}
    else:
        pullback = Auditor.verify_pullback(history, cycle_data['proof'])
    
    if not pullback.get('faithful'):
        return {"id": cid, "status": "integrity_failed"}

    # Blueprinting
    blueprint = query_model(MODEL_ARCHITECT, f"HISTORY: {history}\nTASK: {cycle_data['hypothesis']}\nMANDATORY: Output 'ESTIMATED_MEMORY_KIB: <int>'", "Senior Architect.")
    if "ERROR_PERSISTENT" in blueprint:
        return {"id": cid, "status": "architect_timeout"}
        
    n_est_kb = 10240
    n_match = re.search(r"ESTIMATED_MEMORY_KIB[:\s]*(\d+)", blueprint)
    if n_match:
        n_est_kb = max(512, min(2097152, int(n_match.group(1)))) # Sanity Cap

    # Pre-flight Gating
    decision, L_eff_kb = SystemManager.pre_flight_check(n_est_kb, pullback.get('rho', 0.5))

    # FIX #2: REAL SUBDIVISION using model decomposition
    if decision == "subdivide":
        k_star = SystemManager.calculate_optimal_k(n_est_kb, L_eff_kb)
        print(f"[SystemManager] Subdividing into {k_star} tasks...")
        
        decompose_prompt = (
            f"Decompose this hypothesis into exactly {k_star} independent sub-hypotheses for parallel testing. "
            f"Output STRICTLY valid JSON: {{\"sub_hypotheses\": [\"sub1\", \"sub2\", ...]}}\n"
            f"HYPOTHESIS: {cycle_data['hypothesis']}"
        )
        decompose_result = query_model(MODEL_ARCHITECT, decompose_prompt, "Task Decomposition Specialist.")
        if "ERROR_PERSISTENT" in decompose_result:
             return {"id": cid, "status": "decomposition_timeout"}
             
        sub_hypotheses_data = robust_json_parse(decompose_result)
        
        if not sub_hypotheses_data or "sub_hypotheses" not in sub_hypotheses_data:
            return {"id": cid, "status": "decompose_failed"}
            
        sub_results = []
        for i, sub_hyp in enumerate(sub_hypotheses_data["sub_hypotheses"][:k_star]):
            sub_data = {**cycle_data, "id": f"{cid}_sub{i}", "hypothesis": sub_hyp}
            sub_results.append(run_autonomous_cycle(sub_data, reporter, depth=depth + 1))
            
        # FIX #11: Check sub-task results before marking parent success
        n_success = sum(1 for r in sub_results if r and r.get("status") == "success")
        if n_success == 0:
            return {"id": cid, "status": "subdivided_all_failed", "sub_results": sub_results}
            
        return {"id": cid, "status": "subdivided_complete", "sub_results": sub_results, "success_rate": n_success / len(sub_results)}

    if decision == "reject":
        return {"id": cid, "status": "rejected"}

    # Implementation & Execution
    code_raw = query_model(MODEL_PROGRAMMER, f"BLUEPRINT: {blueprint}", "Computational Physicist.")
    if "ERROR_PERSISTENT" in code_raw:
        return {"id": cid, "status": "programmer_timeout"}
        
    # FIX #14: Safe code block extraction
    code = extract_python_code(code_raw)

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "exp.py").write_text(code)

        # FIX #1, #5, #6, #7, #8: Hardened Sandbox
        bwrap_cmd = [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/usr/local", "/usr/local",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/etc", "/etc",
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
            "python3", "/app/exp.py"
        ]
        
        # FIX #7: ulimit -v (virtual) + -m (RSS) + -u (processes) + -t (cpu time)
        ulimit_prefix = f"ulimit -v {int(L_eff_kb)} -m {int(L_eff_kb)} -u 50 -t 60 && "
        
        # FIX #6: shlex.join prevents shell injection from bwrap args
        cmd = ["bash", "-c", ulimit_prefix + shlex.join(bwrap_cmd)]
        
        # FIX #3: Catch subprocess timeout
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            stdout = res.stdout + res.stderr
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or "") + (e.stderr or "") + "\n[TIMEOUT_EXPIRED: process exceeded 120s]"
        except Exception as e:
            stdout = f"\n[EXECUTION_ERROR: {str(e)}]"

        valid = Auditor.semantic_validation(cycle_data['hypothesis'], stdout)
        
        if valid.get("hypothesis_holds"):
            reporter.update_manifold(cycle_data, valid)
            
        return {"id": cid, "status": "success" if valid.get("hypothesis_holds") else "failed"}

def main():
    if not EnvironmentAuditor.run_readiness_report():
        return

    log_file = next((p / "sheaf_multiagent_research_log.md" for p in [Path("."), Path("erledit"), Path("test_code")] if (p / "sheaf_multiagent_research_log.md").exists()), None)
    if not log_file:
        return

    content = log_file.read_text(encoding="utf-8")
    
    # FIX #16: Better splitting regex
    raw_cycles = [c for c in re.split(r'^---\s*$', content, flags=re.MULTILINE) if "**Hypothesis:**" in c]

    reporter, cp_path = KnowledgeReporter(), Path(CHECKPOINT_FILE)
    
    # FIX #4: Safe checkpoint loading
    done_ids = json.loads(cp_path.read_text()).get("done", []) if cp_path.exists() else []
    
    pending_tasks = []
    for raw in raw_cycles:
        # Match "### Cycle X - Title"
        id_match = re.search(r"### Cycle (\d+)\s*-\s*(.*)", raw)
        if id_match:
            # Create a slug-like ID: "1_Sheaf_Cohomology"
            cycle_num = id_match.group(1)
            title_slug = re.sub(r"[^a-zA-Z0-9]+", "_", id_match.group(2).strip())[:30].strip("_")
            cid = f"{cycle_num}_{title_slug}"
        else:
            cid = hashlib.sha256(raw.encode()).hexdigest()[:16]
        
        if cid in done_ids:
            continue
            
        m_hyp = re.search(r"\*\*hypothesis[:\s]*\*\*[:\s]*(.*?)(?=\n\*\*)", raw, re.DOTALL | re.IGNORECASE)
        m_proof = re.search(r"\*\*proof[:\s]*\*\*[:\s]*(.*?)(?=\nverdict|\n\*\*|---|$)", raw, re.DOTALL | re.IGNORECASE)
        
        if not m_hyp:
            continue
            
        data = {"id": cid, "hypothesis": m_hyp.group(1).strip(), "proof": m_proof.group(1).strip() if m_proof else ""}
        pending_tasks.append(data)

    print(f"Batch Start. {len(pending_tasks)} pending.")

    # FIX #17: Use ThreadPoolExecutor for actual parallelism
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool:
        future_to_cid = {pool.submit(run_autonomous_cycle, data, reporter): data["id"] for data in pending_tasks}
        
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
