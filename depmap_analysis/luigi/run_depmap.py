import pandas as pd
from os import path, makedirs
from pathlib import Path
from luigi import Task, LocalTarget, Parameter, ExternalTask
from luigi.contrib.s3 import S3Target

from depmap_analysis.network_functions.depmap_network_functions import \
    raw_depmap_to_corr, merge_corr_df


def get_root() -> Path:
    return Path(__file__).parent.parent


HERE = Path(__file__)
CRISPR_RAW_FNAME = 'Achilles_gene_effect.csv'
CRISPR_CORR_FNAME = '_crispr_all_correlations.h5'
RNAI_RAW_FNAME = 'D2_combined_gene_dep_scores.csv'
RNAI_CORR_FNAME = '_rnai_all_correlations.h5'


class ProcessDM(Task):
    crispr_raw_fname = CRISPR_RAW_FNAME
    crispr_corr_fname = CRISPR_CORR_FNAME
    rnai_raw_fname = RNAI_RAW_FNAME
    rnai_corr_fname = RNAI_CORR_FNAME
    dm_crispr_release = Parameter()
    dm_rnai_release = Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_to_corr_map = {
            self.crispr_raw.as_posix(): self.crispr_corr.as_posix(),
            self.rnai_raw.as_posix(): self.rnai_corr.as_posix()
        }

    def requires(self):
        return InputFiles(dm_crispr_release=self.dm_crispr_release,
                          dm_rnai_release=self.dm_rnai_release)

    def output(self):
        return [LocalTarget(self.crispr_corr.as_posix()),
                LocalTarget(self.rnai_corr.as_posix())]

    def run(self):
        for fpath in self.input():
            corr = raw_depmap_to_corr(pd.read_csv(fpath.path, index_col=0))
            corr.to_hdf(self.raw_to_corr_map[fpath.path], 'correlations')


class InputFiles(ExternalTask):
    crispr_raw_fname = CRISPR_RAW_FNAME
    rnai_raw_fname = RNAI_RAW_FNAME
    input_dir = get_root().joinpath('input_data', 'depmap')
    dm_crispr_release = Parameter()
    dm_rnai_release = Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crispr_raw = self.input_dir.joinpath(self.dm_crispr_release,
                                                  self.crispr_raw_fname)
        self.rnai_raw = self.input_dir.joinpath(self.dm_rnai_release,
                                                self.rnai_raw_fname)

    def output(self):
        return [LocalTarget(self.crispr_raw.as_posix()),
                LocalTarget(self.rnai_raw.as_posix())]
