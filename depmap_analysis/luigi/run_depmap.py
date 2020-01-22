import pandas as pd
from os import path, makedirs
from pathlib import Path
from luigi import Task, LocalTarget
from luigi.contrib.s3 import S3Target

from depmap_analysis.network_functions.depmap_network_functions import \
    raw_depmap_to_corr, merge_corr_df


def get_root() -> Path:
    return Path(__file__).parent.parent


HERE = Path(__file__)


class ProcessDM(Task):
    crispr_raw_fname = 'Achilles_gene_effect.csv'
    crispr_corr_fname = '_crispr_all_correlations.h5'
    rnai_raw_fname = 'D2_combined_gene_dep_scores.csv'
    rnai_corr_fname = '_rnai_all_correlations.h5'

    def __init__(self, dm_crispr_release, dm_rnai_release, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm_crispr_release = dm_crispr_release
        self.dm_rnai_release = dm_rnai_release
        self.input_dir = get_root().joinpath('input_data', 'depmap')
        self.crispr_corr = self.input_dir.joinpath(self.dm_crispr_release,
                                                   self.crispr_corr_fname)
        self.crispr_raw = self.input_dir.joinpath(self.dm_crispr_release,
                                                  self.crispr_raw_fname)
        self.rnai_corr = self.input_dir.joinpath(self.dm_rnai_release,
                                                 self.rnai_corr_fname)
        self.rnai_raw = self.input_dir.joinpath(self.dm_rnai_release,
                                                self.rnai_raw_fname)


class PreProcessDMData(ProcessDM):
    """Merge correlation matrices to a combined z-score"""
    comb_corr_fname = 'combined_z_score.h5'

    def __init__(self, dm_crispr_release, dm_rnai_release, *args, **kwargs):
        super().__init__(dm_crispr_release=dm_crispr_release,
                         dm_rnai_release=dm_rnai_release, *args, **kwargs)
        self.outdir = self.input_dir.joinpath(
            f'{self.dm_crispr_release}_{self.dm_rnai_release}')
        self.output_fpath = self.outdir.joinpath(self.comb_corr_fname)

    def requires(self):
        return DMRaw2Corr(dm_crispr_release=self.dm_crispr_release,
                          dm_rnai_release=self.dm_rnai_release)

    def run(self):
        options = {'corr_df': pd.read_hdf(self.crispr_corr),
                   'other_corr_df': pd.read_hdf(self.rnai_corr)}
        comb_z_sc_df = merge_corr_df(**options)
        comb_z_sc_df.to_hdf(self.output_fpath.as_posix(), 'combined_z_score')

    def output(self):
        if not self.outdir.is_dir():
            makedirs(self.outdir.as_posix())
        return LocalTarget(
            self.output_fpath.as_posix()
        )


class DMRaw2Corr(ProcessDM):
    """Create correlation matrices from raw depmap data"""

    def __init__(self, dm_crispr_release, dm_rnai_release, *args, **kwargs):
        super().__init__(dm_crispr_release=dm_crispr_release,
                         dm_rnai_release=dm_rnai_release, *args, **kwargs)
        self.raw_to_corr_map = {
            self.crispr_raw.as_posix(): self.crispr_corr.as_posix(),
            self.rnai_raw.as_posix(): self.rnai_corr.as_posix()
        }

    def requires(self):
        return [LocalTarget(self.crispr_raw.as_posix()),
                LocalTarget(self.rnai_raw.as_posix())]

    def output(self):
        return [LocalTarget(self.crispr_corr.as_posix()),
                LocalTarget(self.rnai_corr.as_posix())]

    def run(self):
        for fpath in self.requires():
            corr = raw_depmap_to_corr(pd.read_csv(fpath.path, index_col=0))
            corr.to_hdf(self.raw_to_corr_map[fpath.path], 'correlations')
