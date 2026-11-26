"""Run the support-ticket pipeline.

    python main.py                                   # offline default: rules + lexical TF-IDF + template
    python main.py --generator gemini --classifier gemini   # optional LLM (needs GEMINI_API_KEY)
    python main.py --retriever hybrid                # optional MiniLM semantic layer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from support_pipeline.data import InputDataError
from support_pipeline.pipeline import RunOptions, run_pipeline

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Classify tickets, retrieve help-centre articles, draft grounded replies.")
    parser.add_argument("--tickets", type=Path, default=ROOT / "tickets.json")
    parser.add_argument("--kb", type=Path, default=ROOT / "kb_articles.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--classifier", choices=["rules", "gemini"], default="rules")
    parser.add_argument("--retriever", choices=["lexical", "hybrid"], default="lexical")
    parser.add_argument("--generator", choices=["template", "gemini"], default="template")
    args = parser.parse_args(argv)

    try:
        summary = run_pipeline(args.tickets, args.kb, args.out_dir,
                               RunOptions(args.classifier, args.retriever, args.generator))
    except InputDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    c = summary["components"]
    print(f"{summary['tickets']} tickets | {summary['escalated']} escalated | {summary['fallbacks']} fallbacks | "
          f"classifier={c['classifier']} retriever={c['retriever']} generator={c['generator']}")
    for note in summary["notes"]:
        print(f"note: {note}")
    print(f"wrote {args.out_dir / 'results.json'}, debug_report.md, debug_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
