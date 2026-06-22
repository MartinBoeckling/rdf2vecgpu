#!/usr/bin/env python
"""Print the count of NaN/inf cells in (a sample of) an embedding TSV.

Used by kgreat_eval.sbatch to decide whether to quarantine a diverged embedding.
Takes the path as an ARG (not stdin) — `conda run` does not forward heredoc stdin
to `python -`, which silently broke the previous inline check.
"""
import sys
import pandas as pd
import numpy as np

df = pd.read_csv(sys.argv[1], sep="\t", header=None, index_col=0, nrows=200000)
M = df.to_numpy(dtype=float)
print(int(np.isnan(M).sum() + np.isinf(M).sum()))
