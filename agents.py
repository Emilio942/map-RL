import json
import os
import random
import traceback
from ollama import Client


# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_RESEARCHER = "nemotron-cascade-2:latest"
MODEL_CRITIC     = "nemotron-cascade-2:latest"
MODEL_SYNTHESIS  = "nemotron-cascade-2:latest"
MODEL_LOGISTICS  = "nemotron-cascade-2:latest"
# ==============================================================================


class MathematicalResearchSystem:
    def __init__(self, topic):
        self.client = Client(host='http://localhost:11434')
        self.topic = topic
        self.mathematical_state = ""
        self.insight_archive = []
        self.log_file = "neuro_symbolic_research.md"
        self.cluster_history = []
        self.clusters = [
            "AlgebraicGeometry",
            "Analysis",
            "Topology",
            "Logic",
            "NumberTheory",
            "DifferentialGeometry",
            "DynamicalSystems",
            "ProbabilityTheory"
        ]

    def _query(self, model, system, prompt, is_json=True, max_retries=3):
        """
        Kombinierte Version: Sicheres API-Handling (User) + robuster JSON-Schutz (KI)
        """
        for attempt in range(max_retries):
            try:
                options = {"format": "json"} if is_json else {}
                res = self.client.generate(model=model, system=system, prompt=prompt, stream=False, **options)
                
                # --- DER FIX VOM USER: Sicheres Response-Handling ---
                if isinstance(res, dict):
                    content = res.get('response', '')
                elif hasattr(res, 'response'):
                    content = res.response
                else:
                    content = str(res)
                
                if not is_json:
                    return content.strip()

                content_stripped = content.strip()
                
                # --- DER FIX DER KI: Robuste Extraktion ohne gierige Regex ---
                # Versuch 1: Direkter Parse
                try:
                    data = json.loads(content_stripped)
                    if isinstance(data, (dict, list)):
                        return data
                except json.JSONDecodeError:
                    pass
                
                # Versuch 2: Finde den ersten Start- und letzten End-Tag
                start_idx = -1
                end_idx = -1
                
                obj_start = content_stripped.find('{')
                arr_start = content_stripped.find('[')
                
                if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
                    start_idx = obj_start
                    end_idx = content_stripped.rfind('}')
                elif arr_start != -1:
                    start_idx = arr_start
                    end_idx = content_stripped.rfind(']')
                    
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = content_stripped[start_idx:end_idx+1]
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, (dict, list)):
                            return data
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: JSON Extraktion fehlgeschlagen. Error: {e}")
                        pass
                
                print(f"DEBUG: Kein gültiges JSON gefunden (Versuch {attempt + 1}/{max_retries}).")
                
            except Exception as e:
                print(f"DEBUG: KI Query fehlgeschlagen für {model} (Versuch {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    traceback.print_exc()
        
        # Expliziter Fehler-Return
        if is_json:
            print("DEBUG: Alle JSON-Versuche fehlgeschlagen. Returne leeres Dict.")
            return {}
        else:
            return "Error in mathematical process."

    def _levenshtein(self, s1, s2):
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _relative_novelty(self, question, archive):
        if not archive:
            return 1.0
        distances = [self._levenshtein(question, a) for a in archive]
        min_dist = min(distances)
        max_len = max(len(question), max(len(a) for a in archive))
        if max_len == 0:
            return 1.0
        return min_dist / max_len

    def _select_cluster(self):
        available = [c for c in self.clusters if c not in self.cluster_history[-3:]]
        if not available:
            available = self.clusters
        return random.choice(available)

    def synthesize(self, mathematical_input):
        system = (
            "You are the Mathematical Synthesis Specialist. "
            "Integrate new findings into the consolidated framework. "
            "MANDATORY: Preserve all core axioms and logic. "
            "Output ONLY the updated framework, no commentary."
        )
        # HINWEIS: Doppelte Backslashes entfernt
        full_input = f"OLD STATE: {self.mathematical_state[:4000]}\nNEW INPUT: {mathematical_input}"
        return self._query(MODEL_SYNTHESIS, system, full_input, is_json=False)

    def verify_hypothesis(self, hypothesis):
        proof_sys = "You are a Professor of Mathematics. Provide a rigorous LaTeX proof. No JSON, no formatting — only pure mathematics."
        proof = self._query(MODEL_CRITIC, proof_sys, hypothesis, is_json=False)
        
        verdict_sys = "Is the following proof valid? Answer ONLY 'valid' or 'invalid'."
        verdict_raw = self._query(MODEL_LOGISTICS, verdict_sys, proof[:500], is_json=False)
        
        if 'valid' in verdict_raw.lower() and 'invalid' not in verdict_raw.lower():
            verdict = 'valid'
        elif 'invalid' in verdict_raw.lower():
            verdict = 'invalid'
        else:
            verdict = 'unknown'
        
        return {'proof': proof, 'verdict': verdict}

    def run_exploration(self, iterations=100):
        print(f"--- Starting Autonomous Research Loop ---")
        self.mathematical_state = f"Research Topic: {self.topic}"

        with open(self.log_file, "a", encoding="utf-8") as f:
            # HINWEIS: Doppelte Backslashes hier überall zu einfachen \n korrigiert
            f.write(f"\n\n# RESEARCH START: {self.topic}\n")
            f.write(f"## INITIAL STATE\n{self.mathematical_state}\n\n---\n")

            for i in range(iterations):
                print(f"\n[Cycle {i+1}/{iterations}] Planning...")
                
                cluster = self._select_cluster()
                self.cluster_history.append(cluster)
                
                plan_sys = (
                    "You are a Research Planner. Your task is to suggest 3 new mathematical research angles. "
                    "Respond ONLY with a JSON object in this exact format: "
                    '{"new_perspectives": [{"angle": "title here", "hypothesis": "description here"}]}'
                )
                
                plan_prompt = (
                    f"Topic: {self.topic}\n"
                    f"Current cluster: {cluster}\n"
                    f"Recent clusters: {', '.join(self.cluster_history[-5:])}\n"
                    f"Previously covered: {len(self.insight_archive)} angles\n"
                    f"Suggest 3 fresh mathematical angles for this topic."
                )
                
                discovery = self._query(MODEL_LOGISTICS, plan_sys, plan_prompt)
                
                perspectives = []
                if isinstance(discovery, dict):
                    perspectives = discovery.get('new_perspectives', [])
                elif isinstance(discovery, list) and len(discovery) > 0:
                    perspectives = discovery
                
                if not perspectives:
                    print("  [!] Planner returned no perspectives. Retrying with simpler prompt...")
                    fallback_sys = "Suggest 3 mathematical research angles as JSON."
                    fallback_prompt = f"Topic: {self.topic}. Give me 3 angles."
                    discovery = self._query(MODEL_LOGISTICS, fallback_sys, fallback_prompt)
                    
                    if isinstance(discovery, dict):
                        perspectives = discovery.get('new_perspectives', [])
                    elif isinstance(discovery, list):
                        perspectives = discovery

                if not perspectives:
                    print(f"  [SKIP] Cycle {i+1} aborted - no valid perspectives")
                    continue

                for p in perspectives:
                    if not isinstance(p, dict):
                        continue
                        
                    angle_name = p.get('angle', '') or p.get('title', '') or 'Unknown'
                    hypothesis = p.get('hypothesis', '') or p.get('description', '') or 'No hypothesis'
                    
                    if not angle_name or not hypothesis:
                        continue
                    
                    novelty_score = self._relative_novelty(angle_name, 
                        [a['angle'] for a in self.insight_archive])
                    
                    if novelty_score < 0.5 and len(self.insight_archive) > 5:
                        print(f"  [SKIP] {angle_name} too similar (novelty: {novelty_score:.2f})")
                        continue
                    
                    print(f"  -> Researching: {angle_name} (novelty: {novelty_score:.2f}, cluster: {cluster})")

                    verify_sys = (
                        "You are a Formal Verification Agent. "
                        "Use rigorous mathematics in LaTeX. "
                        'Output JSON: {"proof": "your proof here", "verdict": "valid or invalid"}'
                    )
                    res = self._query(MODEL_CRITIC, verify_sys, hypothesis)
                    
                    proof = 'No proof generated.'
                    verdict = 'unknown'
                    if isinstance(res, dict):
                        proof = res.get('proof', 'No proof generated.')
                        verdict = res.get('verdict', 'unknown')

                    self.insight_archive.append({
                        "angle": angle_name,
                        "verdict": verdict,
                        "cluster": cluster,
                        "novelty": round(novelty_score, 3)
                    })

                    f.write(f"### Cycle {i+1} - {angle_name}\n")
                    f.write(f"**Cluster:** {cluster}\n")
                    f.write(f"**Hypothesis:** {hypothesis}\n")
                    f.write(f"**Verdict:** {verdict}\n")
                    f.write(f"**Novelty Score:** {novelty_score:.3f}\n")
                    f.write(f"**Proof:**\n{proof}\n\n---\n")
                    f.flush()

                    self.mathematical_state = self.synthesize(proof)

        return self.mathematical_state


if __name__ == "__main__":
    t = "Formal Model Theory of Neuro-Symbolic integration: Constructing a Categorical framework for the mapping between continuous neural manifolds and discrete symbolic logic gates."
    system = MathematicalResearchSystem(t)
    system.run_exploration(iterations=5669)
