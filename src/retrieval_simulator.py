"""
MemInterfere: Retrieval Simulator (Phase 5.3)

Implements three experimental tracks to isolate retrieval vs. planning failures:
  - Track A (Gold Retrieval): Inject only the gold skill(s) into the prompt.
    Isolates planning failures — if the agent still fails, it's a planning problem.
  - Track B (RAG Retrieval): Use sentence-transformers to embed skill names +
    descriptions, retrieve top-K by cosine similarity, and show only those.
    Measures retrieval + planning combined.
  - Track C (Full Context): Show all skills (baseline from Phase 4).

Key metrics per track:
  - Selection accuracy: correct tool chosen
  - Invocation accuracy: correct tool + correct parameters
  - Confidence calibration (ECE)
  - Latency: time to first token, total response time
"""

import json
import os
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

import numpy as np

# Local imports
from metrics import (
    EvalTask, EvalResult, TurnLog, Condition, ErrorType,
    compute_ece, compute_success_rate, compute_partial_credit_rate,
    compute_error_distribution,
)

logger = logging.getLogger(__name__)


# ── Embedding cache ──────────────────────────────────────────────────────────

EMBEDDING_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _ensure_cache_dir():
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)


def _cache_path(library_path: str, model_name: str) -> str:
    """Compute a stable cache path for embeddings of a given library."""
    lib_hash = hashlib.md5(library_path.encode()).hexdigest()[:12]
    model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
    return os.path.join(EMBEDDING_CACHE_DIR, f"skill_embeddings_{lib_hash}_{model_hash}.npz")


def _load_embeddings_from_cache(cache_path: str):
    """Load cached embeddings if they exist. Returns (embeddings, skill_ids) or None."""
    if not os.path.exists(cache_path):
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        return data["embeddings"], list(data["skill_ids"])
    except Exception as e:
        logger.warning(f"Failed to load embedding cache from {cache_path}: {e}")
        return None


def _save_embeddings_to_cache(cache_path: str, embeddings: np.ndarray,
                               skill_ids: list[str]):
    """Save embeddings and skill IDs to cache."""
    _ensure_cache_dir()
    np.savez(cache_path, embeddings=embeddings,
             skill_ids=np.array(skill_ids, dtype=object))
    logger.info(f"Saved embeddings to {cache_path}")


def compute_skill_embeddings(
    skills: list,
    model_name: str = DEFAULT_MODEL,
    library_path: str = "",
    force_recompute: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute (or load from cache) embeddings for all skills in the library.

    Each skill is embedded as: f"{skill.name}: {skill.description}"

    Returns:
        (embeddings_array, skill_ids_list)
        embeddings_array shape: (num_skills, embedding_dim)
    """
    # Try cache first
    cache_key = library_path or "inline_library"
    cache_path = _cache_path(cache_key, model_name)

    if not force_recompute:
        cached = _load_embeddings_from_cache(cache_path)
        if cached is not None:
            embeddings, skill_ids = cached
            if len(skill_ids) == len(skills):
                logger.info(f"Loaded {len(skill_ids)} skill embeddings from cache")
                return embeddings, skill_ids
            else:
                logger.warning(f"Cache has {len(skill_ids)} skills but library has {len(skills)}, recomputing")

    # Compute embeddings
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for Track B (RAG retrieval). "
            "Install it with: pip install sentence-transformers"
        )

    logger.info(f"Computing embeddings for {len(skills)} skills using {model_name}...")
    model = SentenceTransformer(model_name)

    texts = [f"{s.name}: {s.description}" for s in skills]
    skill_ids = [s.skill_id for s in skills]

    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    # Save to cache
    _save_embeddings_to_cache(cache_path, embeddings, skill_ids)

    return embeddings, skill_ids


def compute_task_embeddings(
    tasks: list[EvalTask],
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Compute embeddings for task descriptions."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers is required for Track B")

    model = SentenceTransformer(model_name)
    texts = [t.description for t in tasks]
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


# ── Retrieval functions ───────────────────────────────────────────────────────

def retrieve_gold_skills(
    task: EvalTask,
    all_skills: list,
) -> list:
    """
    Track A: Return only the gold (correct) skill(s) for a task.

    Finds skills matching expected_skill_ids and returns them.
    This isolates planning failures — if the agent still selects wrong,
    it's a planning problem, not a retrieval problem.
    """
    gold_skills = []
    for skill in all_skills:
        if skill.skill_id in task.expected_skill_ids or skill.name in task.expected_skill_ids:
            gold_skills.append(skill)

    if not gold_skills:
        logger.warning(f"No gold skills found for task {task.task_id} "
                       f"(expected: {task.expected_skill_ids})")

    return gold_skills


def retrieve_rag_top_k(
    task: EvalTask,
    all_skills: list,
    skill_embeddings: np.ndarray,
    skill_ids: list[str],
    task_embedding: np.ndarray,
    k: int = 5,
) -> tuple[list, list[float]]:
    """
    Track B: Retrieve top-K skills using cosine similarity.

    Args:
        task: The evaluation task.
        all_skills: Full skill library.
        skill_embeddings: Pre-computed skill embeddings (normalized).
        skill_ids: Skill IDs corresponding to embeddings rows.
        task_embedding: Pre-computed embedding for the task description (normalized).
        k: Number of skills to retrieve.

    Returns:
        (retrieved_skills, similarity_scores) sorted by descending similarity.
    """
    # Compute cosine similarities (embeddings are already normalized)
    similarities = skill_embeddings @ task_embedding

    # Get top-K indices
    top_k_indices = np.argsort(similarities)[::-1][:k]

    retrieved_skills = []
    retrieved_scores = []
    for idx in top_k_indices:
        sid = skill_ids[idx]
        # Find the skill object
        for skill in all_skills:
            if skill.skill_id == sid:
                retrieved_skills.append(skill)
                retrieved_scores.append(float(similarities[idx]))
                break

    return retrieved_skills, retrieved_scores


def retrieve_rag_top_k_batch(
    tasks: list[EvalTask],
    all_skills: list,
    skill_embeddings: np.ndarray,
    skill_ids: list[str],
    task_embeddings: np.ndarray,
    k: int = 5,
) -> list[tuple[list, list[float]]]:
    """
    Batch retrieval for multiple tasks.

    Returns:
        List of (retrieved_skills, similarity_scores) tuples, one per task.
    """
    # Compute similarity matrix: (num_tasks, num_skills)
    sim_matrix = task_embeddings @ skill_embeddings.T

    results = []
    for i, task in enumerate(tasks):
        top_k_indices = np.argsort(sim_matrix[i])[::-1][:k]

        retrieved_skills = []
        retrieved_scores = []
        for idx in top_k_indices:
            sid = skill_ids[idx]
            for skill in all_skills:
                if skill.skill_id == sid:
                    retrieved_skills.append(skill)
                    retrieved_scores.append(float(sim_matrix[i][idx]))
                    break

        results.append((retrieved_skills, retrieved_scores))

    return results


# ── Prompt building for each track ─────────────────────────────────────────────

def build_track_prompt(
    task: EvalTask,
    skills_in_prompt: list,
    track_name: str,
    condition: str = "",
) -> str:
    """
    Build the system prompt for a given track, showing only the specified skills.

    This mirrors multi_model_runner.build_system_prompt but allows filtering
    the skill library to just the retrieved skills.

    Args:
        task: The evaluation task.
        skills_in_prompt: Skills to include in the system prompt.
        track_name: Human-readable track name (e.g., "Gold Retrieval", "RAG K=5").
        condition: Condition name for context.

    Returns:
        System prompt string.
    """
    parts = [
        f"You are an AI assistant with access to a skill library ({track_name}).",
        "Given a task, you must select the most appropriate skill from the library",
        "and describe how you would use it.",
        "",
        "## Skill Library",
        "",
    ]

    for skill in skills_in_prompt:
        params_str = ", ".join(
            f"{k}: {v.get('type', 'any')}"
            for k, v in (skill.parameters if isinstance(skill.parameters, dict) else {}).items()
        )
        staleness_note = " [STALE]" if getattr(skill, 'is_stale', False) else ""
        trap_note = " [WARNING: SUSPICIOUS]" if getattr(skill, 'is_trap', False) else ""
        conflict_note = f" [CONFLICT: {skill.conflict_type}]" if skill.conflict_type != "none" else ""

        parts.append(
            f"- **{skill.name}** (id: {skill.skill_id}): {skill.description}"
            f"{staleness_note}{trap_note}{conflict_note}"
            f"\n  Parameters: {params_str}"
        )

    parts.extend([
        "",
        "## Instructions",
        "",
        "Given a task description, respond with EXACTLY this format:",
        "",
        "TOOL: <skill_name>",
        "CONFIDENCE: <number between 0 and 1>",
        "APPROACH: <brief description of how you would use the tool>",
        "",
        "Alternatively, you can respond in JSON format:",
        '```json',
        '{"tool": "<skill_name>", "confidence": <number>, "approach": "<description>"}',
        '```',
        "",
        "IMPORTANT: Choose the most appropriate skill from the library above. "
        "Be careful with skills marked [STALE], [WARNING], or [CONFLICT].",
    ])

    return "\n".join(parts)


# ── RetrievalResult ───────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Result of retrieval for a single task under a specific track."""
    task_id: str
    track: str  # "gold", "rag_k1", "rag_k3", "rag_k5", "rag_k10", "full"
    condition: str
    retrieved_skill_ids: list[str] = field(default_factory=list)
    retrieved_skill_names: list[str] = field(default_factory=list)
    similarity_scores: list[float] = field(default_factory=list)
    gold_skill_ids: list[str] = field(default_factory=list)
    gold_in_top_k: bool = False  # Whether ALL gold skills are in retrieved set
    any_gold_in_top_k: bool = False  # Whether ANY gold skill is in retrieved set
    retrieval_precision: float = 0.0  # fraction of retrieved that are gold
    retrieval_recall: float = 0.0  # fraction of gold that are retrieved


def evaluate_retrieval(
    tasks: list[EvalTask],
    all_skills: list,
    model_name: str = DEFAULT_MODEL,
    library_path: str = "",
    k_values: list[int] = None,
    force_recompute: bool = False,
) -> dict:
    """
    Evaluate retrieval quality across all tracks.

    Returns a dict keyed by track name, each containing:
      - list of RetrievalResult objects
      - aggregate metrics (precision, recall, gold_in_top_k_rate)
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    # Compute embeddings
    skill_embeddings, skill_ids = compute_skill_embeddings(
        all_skills, model_name=model_name,
        library_path=library_path, force_recompute=force_recompute
    )
    task_embeddings = compute_task_embeddings(tasks, model_name=model_name)

    results = {}

    # ── Track A: Gold Retrieval ────────────────────────────────────────────
    gold_results = []
    for i, task in enumerate(tasks):
        gold_skills = retrieve_gold_skills(task, all_skills)
        gold_skill_ids = [s.skill_id for s in gold_skills]
        gold_skill_names = [s.name for s in gold_skills]

        rr = RetrievalResult(
            task_id=task.task_id,
            track="gold",
            condition="gold_retrieval",
            retrieved_skill_ids=gold_skill_ids,
            retrieved_skill_names=gold_skill_names,
            similarity_scores=[1.0] * len(gold_skills),
            gold_skill_ids=task.expected_skill_ids,
            gold_in_top_k=True,  # By definition, gold skills are all present
            any_gold_in_top_k=True,
            retrieval_precision=1.0,  # All retrieved are gold
            retrieval_recall=1.0,    # All gold are retrieved
        )
        gold_results.append(rr)

    results["gold"] = {
        "results": gold_results,
        "avg_retrieval_precision": 1.0,
        "avg_retrieval_recall": 1.0,
        "gold_in_top_k_rate": 1.0,
    }

    # ── Track B: RAG Retrieval at each K ──────────────────────────────────
    for k in k_values:
        track_name = f"rag_k{k}"
        rag_results = []

        # Batch retrieval
        batch = retrieve_rag_top_k_batch(
            tasks, all_skills, skill_embeddings, skill_ids, task_embeddings, k=k
        )

        for i, task in enumerate(tasks):
            retrieved_skills, scores = batch[i]
            retrieved_ids = [s.skill_id for s in retrieved_skills]
            retrieved_names = [s.name for s in retrieved_skills]

            # Compute precision and recall
            gold_set = set(task.expected_skill_ids)
            retrieved_set = set(retrieved_ids)
            # Also match by name
            gold_names = set(task.expected_skill_ids)

            # Check matches by skill_id or name
            matched_gold = set()
            for gs in task.expected_skill_ids:
                for rs in retrieved_skills:
                    if rs.skill_id == gs or rs.name == gs:
                        matched_gold.add(gs)

            gold_in_top_k = len(matched_gold) == len(task.expected_skill_ids)
            any_gold = len(matched_gold) > 0

            precision = len(matched_gold) / len(retrieved_ids) if retrieved_ids else 0.0
            recall = len(matched_gold) / len(task.expected_skill_ids) if task.expected_skill_ids else 0.0

            rr = RetrievalResult(
                task_id=task.task_id,
                track=track_name,
                condition=f"rag_retrieval_k{k}",
                retrieved_skill_ids=retrieved_ids,
                retrieved_skill_names=retrieved_names,
                similarity_scores=scores,
                gold_skill_ids=task.expected_skill_ids,
                gold_in_top_k=gold_in_top_k,
                any_gold_in_top_k=any_gold,
                retrieval_precision=precision,
                retrieval_recall=recall,
            )
            rag_results.append(rr)

        # Aggregate metrics
        avg_precision = np.mean([r.retrieval_precision for r in rag_results])
        avg_recall = np.mean([r.retrieval_recall for r in rag_results])
        gold_rate = np.mean([1.0 if r.gold_in_top_k else 0.0 for r in rag_results])
        any_gold_rate = np.mean([1.0 if r.any_gold_in_top_k else 0.0 for r in rag_results])

        results[track_name] = {
            "results": rag_results,
            "avg_retrieval_precision": float(avg_precision),
            "avg_retrieval_recall": float(avg_recall),
            "gold_in_top_k_rate": float(gold_rate),
            "any_gold_in_top_k_rate": float(any_gold_rate),
            "k": k,
        }

    return results


# ── Track-specific prompt builders for integration ─────────────────────────────

def get_skills_for_track(
    task: EvalTask,
    all_skills: list,
    track: str,
    skill_embeddings: np.ndarray = None,
    skill_ids: list[str] = None,
    task_embedding: np.ndarray = None,
    model_name: str = DEFAULT_MODEL,
) -> tuple[list, list[float]]:
    """
    Get the skills that should be shown in the prompt for a given track.

    Args:
        task: The evaluation task.
        all_skills: Full skill library.
        track: One of "gold", "rag_k1", "rag_k3", "rag_k5", "rag_k10", "full".
        skill_embeddings: Pre-computed skill embeddings (for RAG tracks).
        skill_ids: Skill IDs corresponding to embeddings.
        task_embedding: Pre-computed task embedding (for RAG tracks).
        model_name: Sentence-transformers model name.

    Returns:
        (skills_to_show, similarity_scores)
    """
    if track == "gold":
        skills = retrieve_gold_skills(task, all_skills)
        return skills, [1.0] * len(skills)

    elif track == "full":
        return all_skills, [1.0] * len(all_skills)

    elif track.startswith("rag_k"):
        k = int(track.split("k")[1])
        if skill_embeddings is None or skill_ids is None or task_embedding is None:
            raise ValueError("skill_embeddings, skill_ids, and task_embedding are required for RAG tracks")

        return retrieve_rag_top_k(
            task, all_skills, skill_embeddings, skill_ids,
            task_embedding, k=k
        )

    else:
        raise ValueError(f"Unknown track: {track}. Use 'gold', 'rag_k1', 'rag_k3', 'rag_k5', 'rag_k10', or 'full'")


def get_condition_for_track(track: str) -> str:
    """Map a track name to its Condition enum value."""
    mapping = {
        "gold": "gold_retrieval",
        "rag_k1": "rag_retrieval_k1",
        "rag_k3": "rag_retrieval_k3",
        "rag_k5": "rag_retrieval_k5",
        "rag_k10": "rag_retrieval_k10",
        "full": "all_memory",
    }
    return mapping.get(track, "all_memory")


# ── Main: run retrieval evaluation ────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Set up path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from interference_library import Skill
    from expanded_tasks import ALL_TASKS
    from evaluate_agent import _load_skills

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("MemInterfere Phase 5.3: Retrieval Simulator")
    print("=" * 60)

    # Load skill library (now defaults to expanded 100-skill library)
    skill_library = _load_skills()
    print(f"\nLoaded {len(skill_library)} skills from library")

    # Also load the expanded 100-skill library explicitly for retrieval evaluation
    lib_100_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'skills', 'expanded_library_100.json')
    skill_library_100 = None
    if os.path.exists(lib_100_path):
        with open(lib_100_path) as f:
            data = json.load(f)
        skill_library_100 = [
            Skill(**{k: v for k, v in s.items() if k in Skill.__dataclass_fields__})
            for s in data['skills']
        ]
        print(f"Loaded {len(skill_library_100)} skills from 100-skill library")

    # Run retrieval evaluation
    tasks = ALL_TASKS
    print(f"\nRunning retrieval evaluation on {len(tasks)} tasks...")

    # Evaluate with 68-skill library
    results_68 = evaluate_retrieval(tasks, skill_library)
    print("\n--- 68-Skill Library Retrieval Results ---")
    for track_name, track_data in results_68.items():
        k_info = f" (K={track_data.get('k', 'N/A')})" if 'k' in track_data else ""
        print(f"\n  Track {track_name}{k_info}:")
        print(f"    Avg Precision: {track_data['avg_retrieval_precision']:.4f}")
        print(f"    Avg Recall: {track_data['avg_retrieval_recall']:.4f}")
        print(f"    Gold-in-top-K rate: {track_data['gold_in_top_k_rate']:.4f}")
        if 'any_gold_in_top_k_rate' in track_data:
            print(f"    Any-gold-in-top-K rate: {track_data['any_gold_in_top_k_rate']:.4f}")

    # Evaluate with 100-skill library if available
    if skill_library_100:
        results_100 = evaluate_retrieval(tasks, skill_library_100,
                                          library_path=lib_100_path)
        print("\n--- 100-Skill Library Retrieval Results ---")
        for track_name, track_data in results_100.items():
            k_info = f" (K={track_data.get('k', 'N/A')})" if 'k' in track_data else ""
            print(f"\n  Track {track_name}{k_info}:")
            print(f"    Avg Precision: {track_data['avg_retrieval_precision']:.4f}")
            print(f"    Avg Recall: {track_data['avg_retrieval_recall']:.4f}")
            print(f"    Gold-in-top-K rate: {track_data['gold_in_top_k_rate']:.4f}")
            if 'any_gold_in_top_k_rate' in track_data:
                print(f"    Any-gold-in-top-K rate: {track_data['any_gold_in_top_k_rate']:.4f}")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'results', 'phase5_3')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"retrieval_eval_{timestamp}.json")

    serializable = {}
    for track_name, track_data in results_68.items():
        serializable[track_name] = {
            "avg_retrieval_precision": track_data["avg_retrieval_precision"],
            "avg_retrieval_recall": track_data["avg_retrieval_recall"],
            "gold_in_top_k_rate": track_data["gold_in_top_k_rate"],
            "k": track_data.get("k"),
            "results": [asdict(r) for r in track_data["results"]],
        }

    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "num_tasks": len(tasks),
                "num_skills": len(skill_library),
                "model_name": DEFAULT_MODEL,
            },
            "results": serializable,
        }, f, indent=2)

    print(f"\nSaved retrieval evaluation results to {output_path}")