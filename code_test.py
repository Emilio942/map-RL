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
VENV_PATH = "/home/emilio/Documents/ai/map-RL/.ollama/.venv" # .venv 
MODEL_ARCHITECT = "nemotron-cascade-2:latest"
MODEL_PROGRAMMER = "VibeThinker-3B"#test               "qwen3-coder:30b"
MODEL_AUDITOR = "nemotron-cascade-2:latest"
CHECKPOINT_FILE = "experiment_checkpoint.json"
RESULTS_DIR = Path("results")
MAX_RETRIES = 3
MAX_SUBDIVISION_DEPTH = 3
MAX_SUBDIVISIONS_PER_TASK = 8
OLLAMA_TIMEOUT = 3000
CONCURRENT_REQUESTS = 1

client = Client(host=OLLAMA_HOST)
RESULTS_DIR.mkdir(exist_ok=True)
ollama_semaphore = threading.Semaphore(CONCURRENT_REQUESTS)

# ==============================================================================
# UTILITIES: Robust Logic & Atomic I/O
# ==============================================================================
def query_model(model, prompt, system="", timeout=OLLAMA_TIMEOUT):
    """Thread-safe query with true timeout enforcement and guaranteed semaphore release."""
    print(f"[DEBUG] Querying model {model}...")
    for attempt in range(MAX_RETRIES):
        # 🔧 FIX: Semaphore timeout should be short (waiting for slot, not for model)
        acquired = ollama_semaphore.acquire(timeout=120)
        if not acquired:
            print(f"[Model] Semaphore timeout on {model} (Attempt {attempt+1})")
            continue
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                print(f"[DEBUG] Submitting to executor for {model} (Attempt {attempt+1})...")
                future = executor.submit(client.generate, model=model, prompt=prompt, system=system, stream=False)
                res = future.result(timeout=timeout)
                print(f"[DEBUG] Received response from {model}.")
                return res.get('response', '') if isinstance(res, dict) else getattr(res, 'response', '')
        except TimeoutError:
            print(f"[Model] Timeout on {model} (Attempt {attempt+1})")
            future.cancel()
        except Exception as e:
            print(f"[Model] Error: {e} (Attempt {attempt+1})")
        finally:
            ollama_semaphore.release()
        time.sleep(2 ** attempt)
    return "ERROR_PERSISTENT"


def robust_json_parse(text):
    """Surgically extracts JSON. Uses ast.literal_eval as a fallback for single-quoted dicts."""
    if not text or "ERROR_PERSISTENT" in text:
        text = text.split("<arg_key>output}<arg_value>")[-1]
    elif "</thought>" in text:
        text = text.split("</thought>")[-1]
    else:
        text = re.sub(r"aleigh.*?</tool_call>", "", text, flags=re.DOTALL)
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)

    # Strip markdown code blocks if present around the JSON
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text.strip())

    try:
        start = text.rfind('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            json_str = text[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            start = text.find('[')
            end = text.rfind(']')
        if start == -1 or end == -1:
            return None
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
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
    """Extracts the first ```python ... ``` block, or falls back to heuristic."""
    if not text:
        return ""

    text = re.sub(r"aleigh.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)

    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith(("import ", "from ", "def ", "class ", "print(", "#")):
            print(f"[INFO] Found code-like start at line {i}. Using remaining text as code.")
            return '\n'.join(lines[i:]).strip()

    # 🔧 FIX: Return empty string instead of raw garbage text
    # If no Python is found, code = "" which we catch before execution
    print(f"[WARN] No python code fence or heuristic match found in model output. "
          f"Raw text (first 300 chars): {text[:300]}...")
    return ""  # ← Was: return text.strip()  — executing non-Python is always wrong


def atomic_write_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as tf:
        json.dump(data, tf, indent=2)
        temp_name = tf.name
    os.replace(temp_name, path)


# ==============================================================================
# ENVIRONMENT & SYSTEM MANAGEMENT
# ==============================================================================
class EnvironmentAuditor:
    REQUIRED_LIBS = ["numpy", "scipy", "matplotlib", "sympy", "gudhi", "ripser", "jax", "jax_md", "networkx", "psutil", "ollama"]

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
    K_MAX = MAX_SUBDIVISIONS_PER_TASK
    D_MAX = MAX_SUBDIVISION_DEPTH
    N_STAR_COEFFICIENT = K_MAX ** D_MAX  # 512

    @staticmethod
    def pre_flight_check(n_est_kb, confidence_rho):
        Sv_kb = psutil.virtual_memory().available // 1024
        sigma_buffer = 0.15
        kappa = 1.0 + 0.5 * (1.0 - confidence_rho)
        L_eff_kb = (Sv_kb / kappa) * (1.0 - sigma_buffer)
        delta = n_est_kb - L_eff_kb

        n_star = SystemManager.N_STAR_COEFFICIENT * L_eff_kb if L_eff_kb > 0 else float('inf')
        if n_est_kb > n_star:
            return "reject", L_eff_kb
        if confidence_rho < 0.35:
            return "reject", L_eff_kb
        if delta <= 0 and confidence_rho >= 0.65:
            return "execute", L_eff_kb
        if delta > 0.20 * L_eff_kb:
            return "reject", L_eff_kb
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
            try:
                return json.loads(self.path.read_text())
            except Exception:
                return {}
        return {}

    def update_manifold(self, cycle_data, semantic_val):
        cid = cycle_data['id']
        summary = query_model(MODEL_AUDITOR,
                              f"Summarize Cycle {cid}. Result: {semantic_val.get('reason', 'N/A')}.",
                              "Registrar.")
        if "ERROR_PERSISTENT" in summary:
            summary = "Summary generation failed."
        with self._lock:
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
# PIPELINE
# ==============================================================================
class Auditor:
    @staticmethod
    def verify_pullback(history, proof):
        prompt = (f"ORIGINAL: {proof}\nRETRIEVED: {history}\n"
                  f"Compare logically. Return JSON: {{\"faithful\": bool, \"rho\": float}}")
        raw_output = query_model(MODEL_AUDITOR, prompt, "Security Auditor.")
        parsed = robust_json_parse(raw_output)
        if not parsed:
            print(f"\n[DIAGNOSTICS] JSON parsing failed or empty. Raw LLM Output:\n{raw_output}\n")
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
    if feedback:
        print(f"[DEBUG] Retrying with feedback: {feedback[:100]}...")
    
    history = reporter.get_context()

    if len(history.strip().split('\n')) <= 1:
        pullback = {"faithful": True, "rho": 0.8}
    else:
        print(f"[DEBUG] Verifying pullback for Cycle {cid}...")
        pullback = Auditor.verify_pullback(history, cycle_data['proof'])
    
    if not pullback.get('faithful'):
        print(f"[ERROR] Integrity failed for Cycle {cid}. Pullback not faithful.")
        return {"id": cid, "status": "integrity_failed"}

    print(f"[DEBUG] Generating blueprint for Cycle {cid}...")
    blueprint_prompt = (
        f"HISTORY:\n{history}\n\n"
        f"TASK / HYPOTHESIS:\n{cycle_data['hypothesis']}\n\n"
        f"PREVIOUS FAILURE FEEDBACK (if any):\n{feedback}\n\n"
        f"You are designing a computational experiment. Provide a DETAILED blueprint containing:\n"
        f"1. ALGORITHM: Step-by-step algorithm design and pseudocode\n"
        f"2. DATA STRUCTURES: Key data structures and their relationships\n"
        f"3. LIBRARIES: Required Python libraries (GUDHI, numpy, sympy, networkx)\n"
        f"4. METHOD: How the hypothesis will be computationally tested\n"
        f"5. OUTPUT: Exact print statements the script should produce\n\n"
        f"After your full blueprint, you MUST include this exact footer line "
        f"(required by our orchestration system):\n"
        f"ESTIMATED_MEMORY_KIB: <integer>\n"
        f"where <integer> is your estimate of peak memory usage in KiB."
    )
    blueprint = query_model(MODEL_ARCHITECT, blueprint_prompt, "Senior Architect.")
    if "ERROR_PERSISTENT" in blueprint:
        print(f"[ERROR] Architect timeout for Cycle {cid}.")
        return {"id": cid, "status": "architect_timeout"}

    print(f"[DEBUG] Blueprint generated. Parsing memory estimate...")
    # 🔧 FIX #2: Validate blueprint has actual content beyond just the estimate
    blueprint_lines = [l for l in blueprint.strip().split('\n')
                       if l.strip() and not l.strip().startswith("ESTIMATED_MEMORY_KIB")]
    if len(blueprint_lines) < 3:
        print(f"[WARN] Blueprint too sparse ({len(blueprint_lines)} substantive lines). "
              f"Rejecting cycle {cid}.")
        return {"id": cid, "status": "blueprint_too_sparse"}

    n_est_kb = 10240
    n_match = re.search(r"ESTIMATED_MEMORY_KIB[:\s]*(\d+)", blueprint)
    if n_match:
        n_est_kb = max(512, min(2097152, int(n_match.group(1))))

    decision, L_eff_kb = SystemManager.pre_flight_check(n_est_kb, pullback.get('rho', 0.5))
    print(f"[DEBUG] SystemManager decision: {decision} (L_eff_kb: {L_eff_kb})")

    if decision == "subdivide":
        k_star = SystemManager.calculate_optimal_k(n_est_kb, L_eff_kb)
        print(f"[SystemManager] Subdividing into {k_star} tasks...")
        decompose_prompt = (
            f"Decompose this hypothesis into exactly {k_star} independent sub-hypotheses "
            f"for parallel testing. "
            f"Output STRICTLY valid JSON: {{\"sub_hypotheses\": [\"sub1\", \"sub2\", ...]}}\n"
            f"HYPOTHESIS: {cycle_data['hypothesis']}"
        )
        decompose_result = query_model(MODEL_ARCHITECT, decompose_prompt, "Task Decomposition Specialist.")
        if "ERROR_PERSISTENT" in decompose_result:
            print(f"[ERROR] Decomposition timeout for Cycle {cid}.")
            return {"id": cid, "status": "decomposition_timeout"}
        sub_hypotheses_data = robust_json_parse(decompose_result)
        if not sub_hypotheses_data or "sub_hypotheses" not in sub_hypotheses_data:
            return {"id": cid, "status": "decompose_failed"}

        sub_results = []
        for i, sub_hyp in enumerate(sub_hypotheses_data["sub_hypotheses"][:k_star]):
            sub_data = {**cycle_data, "id": f"{cid}_sub{i}", "hypothesis": sub_hyp}
            sub_results.append(run_autonomous_cycle(sub_data, reporter, depth=depth + 1))

        n_success = sum(1 for r in sub_results if r and r.get("status") == "success")
        if n_success == 0:
            return {"id": cid, "status": "subdivided_all_failed", "sub_results": sub_results}
        return {"id": cid, "status": "subdivided_complete", "sub_results": sub_results,
                "success_rate": n_success / len(sub_results)}

    if decision == "reject":
        print(f"[DEBUG] Cycle {cid} rejected by SystemManager.")
        return {"id": cid, "status": "rejected"}

    # ==========================================================================
    # 🔧 FIX #3 (CRITICAL): Programmer gets hypothesis + blueprint, not just blueprint
    # ==========================================================================
    print(f"[DEBUG] Generating Python code for Cycle {cid}...")
    programmer_prompt = (
        f"You are a Computational Physicist. Write a COMPLETE, SELF-CONTAINED Python script.\n\n"
        f"HYPOTHESIS TO TEST:\n{cycle_data['hypothesis']}\n\n"
        f"IMPLEMENTATION BLUEPRINT:\n{blueprint}\n\n"
        f"PREVIOUS FAILURE FEEDBACK (if any):\n{feedback}\n\n"
        f"MANDATORY REQUIREMENTS:\n"
        f"- Output the code inside a ```python code block\n"
        f"- The script MUST be runnable as `python3 script.py`\n"
        f"- Import ALL needed libraries at the top\n"
        f"- DO NOT USE PLACEHOLDERS, MOCK DATA, OR DUMMY FUNCTIONS. The mathematical "
        f"computation (e.g., cohomology, persistent homology, tensor operations) "
        f"MUST be actually implemented and executed.\n"
        f"- If a library like GUDHI is requested, use its actual API to compute "
        f"dimensions, don't just return hardcoded values.\n"
        f"- Define all functions and classes\n"
        f"- Execute the experiment and print results\n"
        f"- End by printing exactly one of:\n"
        f"  HYPOTHESIS_CONFIRMED: True -- <reason>\n"
        f"  HYPOTHESIS_CONFIRMED: False -- <reason>\n"
        f"- Use only libraries available in standard scientific Python (numpy, sympy, scipy, "
        f"networkx, matplotlib). Avoid jax, gudhi, ripser unless essential."
    )
    code_raw = query_model(MODEL_PROGRAMMER, programmer_prompt, "Computational Physicist.")
    if "ERROR_PERSISTENT" in code_raw:
        print(f"[ERROR] Programmer timeout for Cycle {cid}.")
        return {"id": cid, "status": "programmer_timeout"}

    print(f"[DEBUG] Extracting Python code...")
    code = extract_python_code(code_raw)

    # 🔧 FIX: Save raw output for debugging if extraction fails
    if not code:
        result_dir = RESULTS_DIR / cid
        result_dir.mkdir(parents=True, exist_ok=True)
        dump_path = result_dir / "failed_output_raw.txt"
        dump_path.write_text(code_raw)
        print(f"[ERROR] No valid Python code extracted for cycle {cid}. Raw output saved to {dump_path}")
        return {"id": cid, "status": "no_code_extracted"}

    # 🔧 FIX #5: Secondary validation — code must contain at least one Python keyword
    if not any(kw in code for kw in ["import ", "from ", "def ", "class "]):
        print(f"[ERROR] Extracted code lacks any Python keywords. Aborting execution.")
        return {"id": cid, "status": "invalid_code_structure"}

    # 🔧 FIX #6: Save the code for debugging/audit before execution
    result_dir = RESULTS_DIR / cid
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Saving blueprint and code to {result_dir}...")
    (result_dir / "blueprint.md").write_text(blueprint)
    (result_dir / "code.py").write_text(code)

    print(f"[DEBUG] Executing code in bubblewrap...")
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "exp.py").write_text(code)

        bwrap_cmd = [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/etc", "/etc",
            "--ro-bind", VENV_PATH, VENV_PATH,  # 🔧 The Bridge to your Libraries
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
            f"{VENV_PATH}/bin/python3", "/app/exp.py"
        ]

        # 🔧 FIX #7: Increase process limit to 2048 (bwrap needs more for namespaces)
        ulimit_prefix = f"ulimit -v {int(L_eff_kb)} -m {int(L_eff_kb)} -u 2048 -t 60 && "
        cmd = ["bash", "-c", ulimit_prefix + shlex.join(bwrap_cmd)]

        print(f"[DEBUG] Running bwrap command for Cycle {cid}...")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            stdout = res.stdout + res.stderr
            print(f"[DEBUG] Execution finished. Stdout length: {len(stdout)}")
            if res.returncode != 0:
                print(f"[WARN] Execution returned non-zero exit code: {res.returncode}")
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or "") + (e.stderr or "") + "\n[TIMEOUT_EXPIRED: process exceeded 120s]"
            print(f"[WARN] Execution timed out for Cycle {cid}.")
        except Exception as e:
            stdout = f"\n[EXECUTION_ERROR: {str(e)}]"
            print(f"[ERROR] Execution failed for Cycle {cid}: {e}")

    # 🔧 FIX #8: Save execution output for debugging
    (result_dir / "stdout.txt").write_text(stdout[:50000])

    print(f"[DEBUG] Performing semantic validation...")
    valid = Auditor.semantic_validation(cycle_data['hypothesis'], stdout)
    print(f"[DEBUG] Semantic validation result: {valid.get('hypothesis_holds')} ({valid.get('reason')})")
    
    if valid.get("hypothesis_holds"):
        reporter.update_manifold(cycle_data, valid)
        return {"id": cid, "status": "success"}
    else:
        # FEEDBACK LOOP: If not successful and we haven't retried too much
        max_retries = 2
        if "retry_count" not in cycle_data:
             cycle_data["retry_count"] = 0
             
        if cycle_data["retry_count"] < max_retries:
            print(f"[DEBUG] Validation failed. Retrying cycle {cid} with feedback (Attempt {cycle_data['retry_count']+1})...")
            cycle_data["retry_count"] += 1
            new_feedback = f"Execution Output:\n{stdout[-2000:]}\n\nValidation Reason:\n{valid.get('reason')}"
            return run_autonomous_cycle(cycle_data, reporter, depth=depth, feedback=new_feedback)
        
        return {"id": cid, "status": "failed", "reason": valid.get("reason")}


def main():
    if not EnvironmentAuditor.run_readiness_report():
        return

    log_file = next(
        (p / "sheaf_multiagent_research_log.md"
         for p in [Path("."), Path("erledit"), Path("test_code")]
         if (p / "sheaf_multiagent_research_log.md").exists()),
        None
    )
    if not log_file:
        return

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

        if cid in done_ids:
            continue

        m_hyp = re.search(r"\*\*hypothesis[:\s]*\*\*[:\s]*(.*?)(?=\n\*\*)", raw,
                          re.DOTALL | re.IGNORECASE)
        m_proof = re.search(r"\*\*proof[:\s]*\*\*[:\s]*(.*?)(?=\nverdict|\n\*\*|---|$)", raw,
                            re.DOTALL | re.IGNORECASE)
        if not m_hyp:
            continue

        data = {
            "id": cid,
            "hypothesis": m_hyp.group(1).strip(),
            "proof": m_proof.group(1).strip() if m_proof else ""
        }
        pending_tasks.append(data)

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
