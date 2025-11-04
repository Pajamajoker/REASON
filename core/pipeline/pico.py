import argparse, json, os
from pipeline.extractor.pico_extractor import extract_pico
from utils.io import qid, write_jsonl, now_iso

def _read(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str, help="Free-text question")
    ap.add_argument("--prompt", default="prompts/pico_prompt.txt")
    ap.add_argument("--outdir", default="artifacts_day1")
    args = ap.parse_args()

    prompt = _read(args.prompt)
    args.question = "In adults with septic shock, does early norepinephrine reduce 28-day mortality versus dopamine?"
    pico, quotes, raw = extract_pico(args.question, prompt)

    # save trace
    qhash = qid(args.question)
    out_jsonl = os.path.join(args.outdir, "pico_traces.jsonl")
    out_pretty = os.path.join(args.outdir, f"pico_{qhash}.json")

    trace_row = {
        "qid": qhash,
        "created_at": now_iso(),
        "question": args.question,
        "pico_raw": raw,
        "pico_valid": json.loads(pico.model_dump_json()),
        "quotes": json.loads(quotes.model_dump_json()),
        "model": "gemma-3 via Vertex",
        "temperature": 0.1,
    }
    write_jsonl(out_jsonl, [trace_row])
    with open(out_pretty, "w", encoding="utf-8") as f:
        f.write(json.dumps(trace_row, indent=2))

    # console output
    print("=== PICO ===")
    print(pico.model_dump_json(indent=2))
    print("\n=== QUOTE-BACKS ===")
    print(quotes.model_dump_json(indent=2))
    print(f"\nSaved:\n- {out_jsonl}\n- {out_pretty}")

if __name__ == "__main__":
    main()
