from pathlib import Path

import numpy as np
import pandas as pd
from enigmatoolbox.datasets import load_summary_stats  # type: ignore

if __name__ == "__main__":
    sumstat = load_summary_stats("schizophrenia")["CortThick_case_vs_controls"]
    cohen_d = np.array(sumstat["d_icv"].to_list())
    pvals = sumstat["pobs"]
    rois = sumstat["Structure"]
    df = pd.DataFrame({"roi": rois, "pval": pvals, "cohens_d": cohen_d})
    df.to_csv(Path(__file__).parent / "assets" / "ENIGMA_scz.csv", index=False)
