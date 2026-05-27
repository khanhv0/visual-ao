import json
import random
from pathlib import Path


from instruction_following_eval import instructions_registry


# DATA GENERATION

def make_record(key, base_prompt, instruction_ids, kwargs_list, rng=None):
    """
    Build one IFEval-format record.

    Args:
        key: integer record id.
        base_prompt: the seed prompt (e.g. "Write about renewable energy.").
        instruction_ids: list of IDs from INSTRUCTION_DICT.
        kwargs_list: same length as instruction_ids; per-instruction kwargs.
            Pass {} to let the instruction pick random defaults.
        rng: optional random.Random for reproducibility.

    Returns:
        dict suitable for one line of an IFEval JSONL file.
    """
    if rng is not None:
        random.seed(rng.random())  # the IFEval classes use module-level random

    prompt_parts = [base_prompt.strip()]
    resolved_kwargs = []

    for inst_id, kw in zip(instruction_ids, kwargs_list):
        cls = instructions_registry.INSTRUCTION_DICT[inst_id]
        inst = cls(inst_id)
        description = inst.build_description(**kw)
        prompt_parts.append(description)
        # Capture the kwargs that were actually used (post-defaults)
        resolved_kwargs.append(inst.get_instruction_args() or {})

    return {
        "key": key,
        "prompt": " ".join(prompt_parts),
        "instruction_id_list": list(instruction_ids),
        "kwargs": resolved_kwargs,
    }


def random_record(key, base_prompt, n_instructions=1, rng=None):
    """Pick n random non-conflicting instructions and build a record."""
    rng = rng or random.Random()
    conflicts = instructions_registry.conflict_make(
        {k: set(v) for k, v in instructions_registry.INSTRUCTION_CONFLICTS.items()}
    )
    all_ids = list(instructions_registry.INSTRUCTION_DICT.keys())

    chosen, blocked = [], set()
    for _ in range(n_instructions):
        candidates = [i for i in all_ids if i not in blocked]
        if not candidates:
            break
        pick = rng.choice(candidates)
        chosen.append(pick)
        blocked |= conflicts.get(pick, {pick})

    # Empty kwargs → instructions self-fill with random defaults
    return make_record(key, base_prompt, chosen, [{}] * len(chosen), rng=rng)


def load_base_prompts(source, colname="prompt"):
    """
    Load base prompts from a file or pass-through a list.

    Accepted shapes:
      - list/tuple of strings: used as-is.
      - .txt file: one prompt per line, blank lines skipped.
      - .csv file: reads `colname` column (default 'prompt').
      - .jsonl file: each line is a JSON object; reads `colname` field.
      - None: returns a single empty string, for constraint-only datasets.
    """
    if source is None:
        return [""]
    if isinstance(source, (list, tuple)):
        return [str(p) for p in source]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    if ext == ".txt":
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if ext == ".csv":
        import csv
        with path.open() as f:
            reader = csv.DictReader(f)
            if colname not in reader.fieldnames:
                raise ValueError(
                    f"Column '{colname}' not found in {path}. "
                    f"Available: {reader.fieldnames}"
                )
            return [row[colname].strip() for row in reader if row[colname].strip()]
    if ext == ".jsonl":
        out = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if colname not in obj:
                raise ValueError(f"Field '{colname}' not found in {path}")
            out.append(str(obj[colname]).strip())
        return out

    raise ValueError(f"Unsupported file type: {ext}. Use .txt, .csv, or .jsonl.")


def build_dataset(
    source,
    out_path,
    n_instructions=1,
    seed=0,
    colname="prompt",
    n_records=None,
):
    """
    Build an IFEval-style JSONL dataset.

    Args:
        source: list of base prompts, OR path to .txt/.csv/.jsonl, OR None
                for a constraint-only dataset (no semantic prompt).
        out_path: where to write the JSONL.
        n_instructions: constraints per record (use conflict table).
        seed: RNG seed for reproducibility.
        colname: column/field name to read from CSV or JSONL inputs.
        n_records: if source is None, how many constraint-only records to
                   make (default 10). Ignored when source has real prompts.
    """
    rng = random.Random(seed)
    base_prompts = load_base_prompts(source, colname=colname)

    # Constraint-only mode: replicate the empty prompt n_records times.
    if source is None:
        base_prompts = base_prompts * (n_records or 10)

    with open(out_path, "w") as f:
        for i, p in enumerate(base_prompts):
            rec = random_record(i, p, n_instructions=n_instructions, rng=rng)
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(base_prompts)} records to {out_path}")


# ---------------------------------------------------------------------------
# 2. EVALUATION
# ---------------------------------------------------------------------------

def score_response(record, response):
    """
    Returns (prompt_level_pass, per_instruction_results).
    per_instruction_results is a list of (id, bool).
    """
    per_inst = []
    for inst_id, kw in zip(record["instruction_id_list"], record["kwargs"]):
        cls = instructions_registry.INSTRUCTION_DICT[inst_id]
        inst = cls(inst_id)
        # build_description must be called before check_following because the
        # checker reads parameters set during build_description.
        inst.build_description(**kw)
        try:
            ok = bool(inst.check_following(response))
        except Exception as e:
            print(f"[warn] {inst_id} raised {e}; counting as fail")
            ok = False
        per_inst.append((inst_id, ok))

    prompt_pass = all(ok for _, ok in per_inst)
    return prompt_pass, per_inst


def _loose_variants(response):
    """IFEval's 'loose' check tries several light transforms before scoring."""
    r = response
    yield r
    yield r.strip()
    # Strip a possible leading line (e.g. "Sure! Here you go:") and a trailing one
    lines = r.strip().split("\n")
    if len(lines) > 1:
        yield "\n".join(lines[1:]).strip()
        yield "\n".join(lines[:-1]).strip()
    # Remove markdown bold/italic markers
    yield r.replace("**", "").replace("*", "")


def score_response_loose(record, response):
    """Pass if ANY light variant of the response passes all checks."""
    for v in _loose_variants(response):
        ok, _ = score_response(record, v)
        if ok:
            return True
    return False


def evaluate(dataset_path, responses):
    """
    Args:
        dataset_path: path to the JSONL produced by build_dataset.
        responses: dict mapping record key -> model response string.

    Returns:
        dict of aggregate metrics.
    """
    n_prompts = 0
    n_prompt_strict = 0
    n_prompt_loose = 0
    n_inst = 0
    n_inst_strict = 0

    with open(dataset_path) as f:
        for line in f:
            rec = json.loads(line)
            n_prompts += 1
            resp = responses.get(rec["key"], "")

            strict_pass, per_inst = score_response(rec, resp)
            loose_pass = score_response_loose(rec, resp)

            n_prompt_strict += int(strict_pass)
            n_prompt_loose += int(loose_pass)
            n_inst += len(per_inst)
            n_inst_strict += sum(1 for _, ok in per_inst if ok)

    return {
        "n_prompts": n_prompts,
        "prompt_strict_acc": n_prompt_strict / max(n_prompts, 1),
        "prompt_loose_acc": n_prompt_loose / max(n_prompts, 1),
        "instruction_strict_acc": n_inst_strict / max(n_inst, 1),
    }
