#!/usr/bin/env python
"""Summarize KGrEAT task results into a backend-comparison table.

Reads every <task_id>.tsv under a result dir (columns: id, task_type, dataset,
entities_total, entities_missing, eval_mode, estimator, estimator_config,
embedding_type, metric, score) and prints, per task, the BEST-estimator score
per embedding model on the KNOWN eval mode (i.e. scored only on gold entities
present in the KG). Both metrics are higher-is-better (accuracy,
neg_root_mean_squared_error), so a plain max over estimators is correct.

Usage:  python analyze_kgreat_results.py <result_dir> [--mode KNOWN|ALL] [--csv out.csv]
"""
import sys
import os
import glob
import argparse
import pandas as pd

SHORT = {
    "rdf2vecgpu-gensim": "gensim",
    "rdf2vecgpu-pytorch": "pytorch",
    "rdf2vecgpu-torchrec-rwa": "torchrec",
}


def short(name: str) -> str:
    if name in SHORT:
        return SHORT[name]
    # sweep variants: rdf2vecgpu-torchrec-b131k-lr05 -> b131k-lr05
    for pre in ("rdf2vecgpu-torchrec-", "rdf2vecgpu-"):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--mode", default="KNOWN", choices=["KNOWN", "ALL"])
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.result_dir, "*.tsv")))
    if not files:
        print(f"no result TSVs in {args.result_dir}", file=sys.stderr)
        sys.exit(1)

    records = []          # (task, metric, cover_str, {emb: best_score})
    embeddings = set()
    for f in files:
        try:
            df = pd.read_csv(f, sep="\t")
        except Exception as e:
            print(f"  !! skip {os.path.basename(f)}: {e}", file=sys.stderr)
            continue
        if df.empty:
            continue
        task = str(df["id"].iloc[0])
        metric = str(df["metric"].iloc[0])
        total = int(df["entities_total"].iloc[0])
        missing = int(df["entities_missing"].iloc[0])
        present = total - missing
        sub = df[df["eval_mode"] == args.mode]
        if sub.empty:
            continue
        best = sub.groupby("embedding_type")["score"].max()
        best = {short(k): v for k, v in best.items()}
        embeddings.update(best.keys())
        records.append((task, metric, f"{present}/{total}", best))

    embeddings = sorted(embeddings)
    records.sort(key=lambda r: r[0])

    # ---- pretty table ----
    tw = max([len(r[0]) for r in records] + [4])
    header = f"{'task':<{tw}}  {'cover':>11}  " + "  ".join(f"{e:>9}" for e in embeddings) + "   metric"
    print(header)
    print("-" * len(header))
    wins = {e: 0 for e in embeddings}
    for task, metric, cover, best in records:
        cells = []
        valid = {e: best[e] for e in embeddings if e in best}
        top = max(valid.values()) if valid else None
        for e in embeddings:
            if e in best:
                v = best[e]
                mark = "*" if (top is not None and abs(v - top) < 1e-9) else " "
                if top is not None and abs(v - top) < 1e-9:
                    wins[e] += 1
                cells.append(f"{v:>+8.4f}{mark}")
            else:
                cells.append(f"{'--':>9}")
        m = metric.replace("neg_root_mean_squared_error", "neg_rmse")
        print(f"{task:<{tw}}  {cover:>11}  " + "  ".join(cells) + f"   [{m}]")

    print("-" * len(header))
    print("best-per-task (*) tally:  " + "   ".join(f"{e}={wins[e]}" for e in embeddings))
    print(f"(eval_mode={args.mode}; both metrics higher-is-better; '*' = best in row)")

    if args.csv:
        out = []
        for task, metric, cover, best in records:
            row = {"task": task, "metric": metric, "cover": cover}
            row.update(best)
            out.append(row)
        pd.DataFrame(out).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
