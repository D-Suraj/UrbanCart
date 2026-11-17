"""Run paid live verification without printing credentials. Usage: --limit 3 or --smoke."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

# The private sibling file is local-only and is never packaged.
load_dotenv(ROOT.parent / "Private/env")
load_dotenv(ROOT / ".env", override=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", default="development")
    parser.add_argument("--consistency", action="store_true")
    parser.add_argument("--ids")
    args = parser.parse_args()
    if args.smoke or args.query:
        from urbancart.config import Settings
        from urbancart.service import Assistant, Ask
        core = Assistant(Settings.from_env(mode="openrouter"))
        for q in ([args.query] if args.query else ["Can I return electronics?", "Is UC-ELE-891071 in stock?", "Where is order UC882140?"]):
            r = core.ask(Ask(query=q, customer_id="DEMO-01"))
            print(json.dumps(r.model_dump(), ensure_ascii=True), flush=True)
    else:
        from urbancart.assessment import run
        run(args.limit, args.split, args.consistency, args.ids.split(",") if args.ids else None)
