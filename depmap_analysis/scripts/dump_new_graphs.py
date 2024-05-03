"""Dumps new indra graphs from the latest sif dump"""
import pandas as pd
import logging
import argparse
from typing import Optional

import depmap_analysis.network_functions.net_functions as nf
from depmap_analysis.util.io_functions import file_path, file_opener
from depmap_analysis.util.aws import get_latest_sif_s3, dump_pickle_to_s3, NETS_PREFIX


logger = logging.getLogger(__name__)

__all__ = [
    "INDRA_MDG",
    "INDRA_DG",
    "INDRA_SNG",
    "INDRA_SEG",
    "INDRA_PBSNG",
    "INDRA_PBSEG",
]

INDRA_MDG = "indranet_multi_digraph.pkl"
INDRA_DG = "indranet_dir_graph.pkl"
INDRA_SNG = "indranet_sign_node_graph.pkl"
INDRA_SEG = "indranet_sign_edge_graph.pkl"
INDRA_PBSNG = "indranet_sign_node_pybel.pkl"
INDRA_PBSEG = "indranet_sign_edge_pybel.pkl"


def dump_new_nets(
    mdg: bool = False,
    dg: bool = False,
    sg: bool = False,
    spbg: bool = False,
    verbosity: int = 0,
    add_mesh_ids: bool = False,
    corr_path: Optional[str] = None,
):
    """Main script function for dumping new networks from latest db dumps"""
    options = dict()

    if corr_path:
        logger.info(f"Loading z-score dataframe from {corr_path}")
        corr_df = file_opener(corr_path)
        options["z_sc_path"] = corr_df

    if add_mesh_ids:
        (df, sif_date), (mid, _) = get_latest_sif_s3(get_mesh_ids=True)
        mid_dict = dict()
        for pair in mid:
            mid_dict.setdefault(pair[0], []).append(pair[1])
        options["mesh_id_dict"] = mid_dict
    else:
        df, sif_date = get_latest_sif_s3()

    options.update(
        {
            "df": df,
            "include_entity_hierarchies": True,
            "verbosity": verbosity,
            "date": sif_date,
        }
    )
    prefix = f"{NETS_PREFIX}/{sif_date}/".replace("//", "/")
    if mdg:
        network = nf.sif_dump_df_to_digraph(graph_type="multidigraph", **options)
        dump_pickle_to_s3(INDRA_MDG, network, prefix=prefix)
    if dg:
        network = nf.sif_dump_df_to_digraph(**options)
        dump_pickle_to_s3(INDRA_DG, network, prefix=prefix)
    if sg:
        network, isng = nf.sif_dump_df_to_digraph(graph_type="signed", **options)
        dump_pickle_to_s3(INDRA_SEG, network, prefix=prefix)
        dump_pickle_to_s3(INDRA_SNG, isng, prefix=prefix)
    if spbg:
        pb_seg, pb_sng = nf.db_dump_to_pybel_sg()
        dump_pickle_to_s3(INDRA_SNG, pb_sng, prefix=prefix)
        dump_pickle_to_s3(INDRA_SEG, pb_seg, prefix=prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dump new networks")
    parser.add_argument(
        "--mdg", help="Dump a new MultiDiGraph", action="store_true", default=False
    )
    parser.add_argument(
        "--dg", help="Dump a new DiGraph", action="store_true", default=False
    )
    parser.add_argument(
        "--sg",
        help="Dump new signed edge and node graphs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pb",
        help="Dump new PyBel signed edge and node graphs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--z-score", type=file_path("h5"), help="Path to a correlation matrix"
    )
    args = parser.parse_args()
    dump_new_nets(
        mdg=args.mdg, dg=args.dg, sg=args.sg, spbg=args.pb, corr_path=args.z_score
    )
