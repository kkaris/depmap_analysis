from depmap_analysis.scripts.depmap_meta import _get_outfile_name, _join


def test_outfile_name():
    graph_types = ["unsigned", "signed"]
    arg_path = "s3://depmap-analysis/output_data/" \
               "21Q2crispr_D2v6_20210126/signed_logp/"
    lo = 5.5
    hi = 20

    for graph_type in graph_types:
        outname = _join(arg_path, graph_type)

        # Test closed interval range
        outfile = _get_outfile_name(outname, lo, hi)
        assert (
            outfile == f'{arg_path}{graph_type}_{str(lo).replace(".", "")}_'
            f'{str(hi).replace(".", "")}.pkl'
        ), outfile

        # Test open interval range
        outfile = _get_outfile_name(outname, lo)
        assert (
            outfile ==
            f'{arg_path}{graph_type}_{str(lo).replace(".", "")}_.pkl'
        ), outfile

        # Test max only
        outfile = _get_outfile_name(outname, hi_sd=hi)
        assert (
            outfile ==
            f'{arg_path}{graph_type}_{str(hi).replace(".", "")}.pkl'
        ), outfile
