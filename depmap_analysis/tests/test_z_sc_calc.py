from pathlib import Path

import numpy as np
import pandas as pd

from depmap_analysis.preprocessing.depmap_preprocessing import run_corr_merge
from depmap_analysis.util.statistics import *
from . import *


def test_merge_corr():
    shape = 50, 10
    a, nan_count_a = _get_raw_w_nan(shape, nan_count=30)
    a.index = [f'{str(cn)} {str(cn)}' for cn in a.index]
    a.columns = [str(cn) for cn in a.columns]

    b, nan_count_b = _get_raw_w_nan(shape, nan_count=30)
    b.index = [f'{str(cn)} {str(cn)}' for cn in b.index]
    b.columns = [str(cn) for cn in b.columns]

    # Get samples
    an = get_n(recalculate=True, data_df=a)
    bn = get_n(recalculate=True, data_df=b)

    # Make corr matrices
    a_corr = a.corr()
    b_corr = b.corr()

    # Get logp
    alog = get_logp(recalculate=True, data_corr=a_corr, data_n=an)
    blog = get_logp(recalculate=True, data_corr=b_corr, data_n=bn)

    # Get z
    az = get_z(recalculate=True, data_logp=alog, data_corr=a_corr)
    bz = get_z(recalculate=True, data_logp=blog, data_corr=b_corr)

    # Merge with stouffer's
    stouffer_merged: pd.DataFrame = (az + bz) / np.sqrt(2)

    # Drop nan's (this is default behavior in run_corr_merge)
    stouffer_merged = stouffer_merged.dropna(
        axis=0, how='all').dropna(axis=1, how='all')

    # Set diagonal to NaN
    np.fill_diagonal(a=stouffer_merged.values, val=np.nan)

    Path('temp').mkdir(exist_ok=True)
    afp = 'temp/a_raw.h5'
    bfp = 'temp/b_raw.h5'
    a.to_hdf(afp, 'test')
    b.to_hdf(bfp, 'test')
    merged: pd.DataFrame = run_corr_merge(
        crispr_raw=afp, rnai_raw=bfp, corr_output_dir="temp"
    )

    # Check inf count
    assert (stouffer_merged.abs() == np.inf).sum().sum() == (
        merged.abs() == np.inf
    ).sum().sum()

    # Check NaN count
    assert pd.isna(stouffer_merged).sum().sum() == pd.isna(merged).sum().sum()

    # Are they the same?
    assert stouffer_merged.equals(merged)
