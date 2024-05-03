import logging
import subprocess
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from itertools import cycle
from typing import Tuple, Union, Dict, Optional, List

try:
    # Py 3.8+
    from typing import Literal
except ImportError:
    # Py 3.7-
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import requests
from networkx import DiGraph, MultiDiGraph
from requests.exceptions import ConnectionError
from tqdm import tqdm

from depmap_analysis.util.aws import get_latest_pa_stmt_dump
from depmap_analysis.util.io_functions import file_opener
from indra.assemblers.english import EnglishAssembler
from indra.assemblers.indranet import IndraNet
from indra.assemblers.indranet.net import default_sign_dict
from indra.assemblers.pybel import PybelAssembler
from indra.assemblers.pybel.assembler import belgraph_to_signed_graph
from indra.belief import load_default_probs
from indra.config import CONFIG_DICT
from indra.databases import get_identifiers_url
from indra.explanation.model_checker.model_checker import \
    signed_edges_to_signed_nodes
from indra.explanation.pathfinding import bfs_search
from indra.ontology.bio import bio_ontology
from indra.statements import Agent, get_statement_by_name, get_all_descendants
from indra.util.statement_presentation import reader_sources

logger = logging.getLogger(__name__)

NP_PRECISION = 10 ** -np.finfo(np.longfloat).precision  # Numpy precision
MIN_WEIGHT = np.longfloat(1e-12)  # Set min weight to 10x precision
INT_PLUS = 0
INT_MINUS = 1
SIGN_TO_STANDARD = {INT_PLUS: '+', '+': '+', 'plus': '+',
                    '-': '-', 'minus': '-', INT_MINUS: '-'}
SIGNS_TO_INT_SIGN = {INT_PLUS: INT_PLUS, '+': INT_PLUS, 'plus': INT_PLUS,
                     '-': INT_MINUS, 'minus': INT_MINUS, INT_MINUS: INT_MINUS,
                     None: None}
REVERSE_SIGN = {INT_PLUS: INT_MINUS, INT_MINUS: INT_PLUS,
                '+': '-', '-': '+',
                'plus': 'minus', 'minus': 'plus'}

# Derived types
GraphTypes = Literal['digraph', 'multidigraph', 'signed', 'signed-expanded',
                     'digraph-signed-types']

# FixMe use the "readers" vs db from indra_db, where is it?
READERS = {'reach', 'trips', 'isi', 'sparser', 'medscan', 'rlimsp', 'eidos',
           'cwms', 'geneways', 'tees', 'hume', 'sofia'}.update(reader_sources)


def _get_smallest_belief_prior():
    def_probs = load_default_probs()
    return min([v for v in def_probs['syst'].values() if v > 0])# +
               #[[v for v in def_probs['rand'].values() if v > 0]])


MIN_BELIEF = _get_smallest_belief_prior()


GRND_URI = None
GILDA_TIMEOUT = False
try:
    GRND_URI = CONFIG_DICT['GILDA_URL']
except KeyError:
    logger.warning('Indra Grounding service not available. Add '
                   'INDRA_GROUNDING_SERVICE_URL to `indra/config.ini`')


def pinger(domain, timeout=1):
    """Returns True if host at domain is responding"""
    return subprocess.run(["ping", "-c", "1", '-w%d' % int(timeout),
                           domain]).returncode == 0


def gilda_pinger():
    """Return True if the gilda service is available"""
    try:
        logger.info('Trying to reach GILDA service again...')
        return requests.post(GRND_URI, json={'text': 'erk'}).status_code == 200
    except ConnectionError:
        return False


def _curated_func(ev_dict):
    """Return False if no source dict exists, or if all sources are
    readers, otherwise return True."""
    return False if not ev_dict or not isinstance(ev_dict, dict) else \
        (False if all(s.lower() in READERS for s in ev_dict) else True)


def _weight_from_belief(belief):
    """Map belief score 'belief' to weight. If the calculation goes below
    precision, return longfloat precision instead to avoid making the
    weight zero."""
    return np.max([MIN_WEIGHT, -np.log(belief, dtype=np.longfloat)])


def _weight_mapping(G, verbosity=0):
    """Mapping function for adding the weight of the flattened edges

    Parameters
    ----------
    G : IndraNet
        Incoming graph

    Returns
    -------
    G : IndraNet
        Graph with updated belief
    """
    with np.errstate(all='raise'):
        for edge in G.edges:
            try:
                G.edges[edge]['weight'] = \
                    _weight_from_belief(G.edges[edge]['belief'])
            except FloatingPointError as err:
                logger.warning('FloatingPointError from unexpected belief '
                               '%s. Resetting ag_belief to 10*np.longfloat '
                               'precision (%.0e)' %
                               (G.edges[edge]['belief'],
                                Decimal(MIN_WEIGHT)))
                if verbosity == 1:
                    logger.error('Error string: %s' % err)
                elif verbosity > 1:
                    logger.error('Exception output follows:')
                    logger.exception(err)
                G.edges[edge]['weight'] = MIN_WEIGHT
    return G


def _english_from_row(row):
    return _english_from_agents_type(row.agA_name, row.agB_name,
                                     row.stmt_type)


def _english_from_agents_type(agA_name, agB_name, stmt_type):
    agA = Agent(agA_name)
    agB = Agent(agB_name)
    StmtClass = get_statement_by_name(stmt_type)
    if stmt_type.lower() == 'complex':
        stmt = StmtClass([agA, agB])
    else:
        stmt = StmtClass(agA, agB)
    return EnglishAssembler([stmt]).make_model()


def expand_signed(df: pd.DataFrame, sign_dict: Dict[str, int],
                  stmt_types: List[str], use_descendants: bool = True) \
        -> pd.DataFrame:
    """Expands out which statements should be added to the signed graph

    The statements types provided in 'stmt_types' will be added for both
    signs. To add more statement types of just one sign, add it to 'sign_dict'.

    Parameters
    ----------
    df : pd.DataFrame
    sign_dict : Dict[str, int]
        A dictionary mapping a Statement type to a sign to be used for the
        edge. By default only Activation and IncreaseAmount are added as
        positive edges and Inhibition and DecreaseAmount are added as
        negative edges, but a user can pass any other Statement types in a
        dictionary.
    stmt_types : List[str]
        The statement types to match to expand signs to. The rows matching
        these types will be duplicated and each copy gets a distinct sign.
    use_descendants : bool
        If True, also match descendants of the statements provided in
        'stmt_types' when adding the extended signs.

    Returns
    -------
    pd.DataFrame
    """
    if use_descendants:
        logger.info('Getting descendants to match for expanded signed graph')
        # Get name of descendants
        more_stmt_types = set(stmt_types)
        for s in stmt_types:
            more_stmt_types.update({
                s.__name__ for s in
                get_all_descendants(get_statement_by_name(s))
            })
        stmt_types = list(more_stmt_types)

    # Add new sign column, set to None. Using 'initial_sign' allows usage of
    # IndraNet.to_signed_graph
    df['initial_sign'] = None

    # Locate relevant rows
    standard_sign = df.stmt_type.isin(sign_dict.keys())
    expand_sign = df.stmt_type.isin(stmt_types)
    assert sum(standard_sign) + sum(expand_sign) > 0, \
        'All rows filtered out from DataFrame. Check that statement types ' \
        'in sign_dict and stmt_types exist in the DataFrame.'
    if sum(expand_sign) == 0:
        logger.warning('No rows can be used for expanded signed edges. Check '
                       'that statement types in stmt_types exist in the '
                       'DataFrame.')

    # Add sign for signed statements
    logger.info('Setting initial sign for signed types')
    df.loc[standard_sign, 'initial_sign'] = \
        df.loc[standard_sign, 'stmt_type'].apply(lambda st: sign_dict.get(st))

    # Add positive sign to the rows with types in stmt_types
    df.loc[expand_sign, 'initial_sign'] = INT_PLUS

    # Copy rows for expand sign and switch sign
    logger.info('Setting initial sign for expanded signed types')
    add_rows = []
    for _, expand_row in df[expand_sign].iterrows():
        exp_row = [INT_MINUS if col == 'initial_sign' else val
                   for col, val in expand_row.items()]
        add_rows.append(exp_row)

    logger.info('Appending extended signed rows')
    extra_df = pd.DataFrame(add_rows, columns=df.columns.values)
    df = df.append(extra_df)

    # Remove all rows without assigned sign
    logger.info('Removing rows without signed')
    df = df[~df.initial_sign.isna()]

    # Re-cast sign column as int
    try:
        df.initial_sign = df.initial_sign.astype(pd.Int32Dtype())
    except Exception as exc:
        link = 'https://pandas.pydata.org/pandas-docs/stable/user_guide' \
          '/integer_na.html'
        logger.warning(f'Could not set sign column as Nullable Integer Data '
                       f'Type. MAke sure to use pandas v0.24+. See {link}')

    return df


def sif_dump_df_merger(df: pd.DataFrame,
                       graph_type: str,
                       sign_dict: Optional[Dict[str, int]] = None,
                       stmt_types: Optional[List[str]] = None,
                       mesh_id_dict: Optional[Dict[str, str]] = None,
                       set_weights: bool = True,
                       verbosity: int = 0):
    """Merge the sif dump df with the provided dictionaries

    Parameters
    ----------
    df : str|pd.DataFrame
        A dataframe, either as a file path to a pickle or csv, or a pandas
        DataFrame object.
    graph_type : str
        If 'signed-expanded' or 'digraph-signed-types', do extra filtering
        or alteration to the DataFrame to produce an expanded signed graph
        or a reduced digraph with only the signed types
    sign_dict : Optional[Dict[str, int]]
        A dictionary mapping a Statement type to a sign to be used for the
        edge. By default only Activation and IncreaseAmount are added as
        positive edges and Inhibition and DecreaseAmount are added as
        negative edges, but a user can pass any other Statement types in a
        dictionary.
    stmt_types : Optional[List[str]]
        Provide a list of statement types to be used if expanding the signed
        graph to include statements of these types
    mesh_id_dict : dict
        A dict object mapping statement hashes to all mesh ids sharing a
        common PMID
    set_weights : bool
        If True, set the edge weights. Default: True.
    verbosity : int
        Output various extra messages if > 1.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with new columns from the merge
    """
    if isinstance(df, str):
        merged_df = file_opener(df)
    else:
        merged_df = df

    if 'hash' in merged_df.columns:
        merged_df.rename(columns={'hash': 'stmt_hash'}, inplace=True)

    # Extend df with these columns:
    #   english string from mock statements
    #   mesh_id mapped by dict (if provided)
    #   z-score values (if provided)
    # Extend df with famplex rows
    # 'stmt_hash' must exist as column in the input dataframe for merge to work
    # Preserve all rows in merged_df, so do left join:
    # merged_df.merge(other, how='left', on='stmt_hash')

    if graph_type == 'signed-expanded' and sign_dict and stmt_types:
        merged_df = expand_signed(merged_df, sign_dict, stmt_types)
    elif graph_type == 'signed-expanded' and not (sign_dict and stmt_types):
        raise ValueError('Must provide statement types using variable '
                         '`stmt_types` to run signed_expanded graph')

    if mesh_id_dict is not None:
        hashes = []
        mesh_ids = []
        for k, v in mesh_id_dict.items():
            hashes.append(int(k))
            mesh_ids.append(v)

        merged_df = merged_df.merge(
            right=pd.DataFrame(data={'stmt_hash': hashes,
                                     'mesh_ids': mesh_ids}),
            how='left',
            on='stmt_hash'
        )

    # Check for missing hashes
    if merged_df['source_counts'].isna().sum() > 0:
        logger.warning('%d rows with missing evidence found' %
                       merged_df['source_counts'].isna().sum())
        if verbosity > 1:
            logger.info(
                'Missing hashes in stratified evidence dict: %s' %
                list(merged_df['stmt_hash'][
                         merged_df['source_counts'].isna() == True]))

    logger.info('Setting "curated" flag')
    # Map to boolean 'curated' for reader/non-reader
    merged_df['curated'] = merged_df['source_counts'].apply(func=_curated_func)

    # Make english statement
    merged_df['english'] = merged_df.apply(_english_from_row, axis=1)

    if set_weights:
        logger.info('Setting edge weights')
        # Add weight: -log(belief) or 1/evidence count if no belief
        has_belief = (merged_df['belief'].isna() == False)
        has_no_belief = (merged_df['belief'].isna() == True)
        merged_df['weight'] = 0
        if has_belief.sum() > 0:
            merged_df.loc[has_belief, 'weight'] = merged_df['belief'].apply(
                func=_weight_from_belief)
        if has_no_belief.sum() > 0:
            merged_df.loc[has_no_belief, 'weight'] = \
                merged_df['evidence_count'].apply(
                    func=lambda ec: 1/np.longfloat(ec))
    else:
        logger.info('Skipping setting belief weight')

    return merged_df


def add_corr_to_edges(graph: DiGraph, z_corr: pd.DataFrame,
                      self_corr_value: Optional[float] = None):
    """Add z-score and associated weight to graph edges

    Parameters
    ----------
    graph :
        The DiGraph to add the edge attributes to
    z_corr :
        A square dataframe with all correlations
    self_corr_value :
        If provided, set this value as self corr value. Default: value of 
        first non-NaN value on the diagonal.
    """
    logger.info('Setting z-scores and z-score weights to graph edges')
    self_corr = None
    if self_corr_value is not None:
        self_corr = self_corr_value
    else:
        for d in z_corr.values.diagonal():
            if not np.isnan(d):
                self_corr = d
                break
    if not isinstance(self_corr, (int, float, np.floating, np.integer)):
        raise ValueError('Provide a value for self correlation or a z-score '
                         'dataframe with self correlations')
    non_z_score = 0
    non_corr_weight = round(
        z_sc_weight(z_score=non_z_score, self_corr=self_corr), 4
    )
    for u, v, data in tqdm(graph.edges(data=True)):
        un = u[0] if isinstance(u, tuple) else u
        vn = v[0] if isinstance(v, tuple) else v
        if (
                un in z_corr and vn in z_corr and
                not np.isnan(z_corr.loc[un, vn]) and
                not np.isinf(z_corr.loc[un, vn])
        ):
            z_sc = z_corr.loc[un, vn]
            data['z_score'] = round(z_sc, 4)
            data['corr_weight'] = round(z_sc_weight(z_sc, self_corr), 4)
        else:
            data['z_score'] = non_z_score
            data['corr_weight'] = non_corr_weight

    logger.info('Performing sanity checks')
    assert all('corr_weight' in graph.edges[e] and 'z_score' in graph.edges[e]
               for e in tqdm(graph.edges)), \
        'Some edges are missing z_score or corr_weight attributes'
    assert all(graph.edges[e]['corr_weight'] > 0 for e in tqdm(graph.edges)), \
        'Some values of corr_weight are <= 0'
    logger.info('Done setting z-scores and z-score weights')


def get_corrs(z_sc_df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Getting available hgnc symbols from correlation matrix')
    corr_symb_set = set(z_sc_df.columns.values)
    logger.info('Stacking the correlation matrix: may take a couple of '
                'minutes and tens of GiB of memory')
    stacked_z_sc_df = z_sc_df.stack(
        dropna=True
    ).to_frame(
        name='z_score',
    ).reset_index().rename(
        columns={'level_0': 'agA_name', 'level_1': 'agB_name'}
    )

    # Merge in stacked correlations to the sif df
    logger.info('Getting relevant correlations')
    z_corr_pairs = merged_df[['agA_name', 'agB_name']].merge(
        right=stacked_z_sc_df, how='left'
    ).drop_duplicates()

    # z_score: original z-score or 0 if nonexistant
    z_corr_pairs.loc[z_corr_pairs.z_score.isna(), 'z_score'] = 0

    # Get self correlation
    self_corr = z_sc_df.iloc[0, 0]
    assert isinstance(self_corr, (int, float)) and self_corr > 0

    # Calculate corr weight = (self_corr_z_sc - abs(z_score)) / self_corr
    z_corr_pairs['corr_weight'] = z_sc_weight_df(z_corr_pairs, self_corr)
    logger.info('Finished setting z-score and z-score weight in sif df')
    return z_corr_pairs


def z_sc_weight_df(df: pd.DataFrame, self_corr: float) -> pd.Series:
    """Calculate the corresponding weight of a z-score from a dataframe

    Parameters
    ----------
    df :
        A dataframe that contains at least the column 'z_score'
    self_corr :
        The self correlation value

    Returns
    -------
    :
        The difference between self_corr and the absolute value of the
        z-score as a series
    """
    # Set z-score weight w = self corr z-score - abs(z-score) / self corr z-score
    out_series: pd.Series = (self_corr - df.z_score.abs()) / self_corr

    # Set self corr values and NaN's to a weight of 1
    out_series[(out_series == 0) | out_series.isna()] = 1

    return out_series


def z_sc_weight(z_score: float, self_corr: float) -> float:
    """Calculate the corresponding weight of a given z-score

    If z_score == self_corr, return self_corr.

    Parameters
    ----------
    z_score:
        The z-score to calculate the weight of
    self_corr:
        The self correlation value

    Returns
    -------
    :
        The difference between self_corr and the absolute value of the
        z-score normalized, unless z_score == self_corr, then return 1
    """
    if self_corr == z_score:
        return 1
    return (self_corr - abs(z_score)) / self_corr


def sif_dump_df_to_digraph(df: Union[pd.DataFrame, str],
                           date: str,
                           mesh_id_dict: Optional[Dict] = None,
                           graph_type: GraphTypes = 'digraph',
                           include_entity_hierarchies: bool = True,
                           sign_dict: Optional[Dict[str, int]] = None,
                           stmt_types: Optional[List[str]] = None,
                           z_sc_path: Optional[Union[str, pd.DataFrame]] = None,
                           verbosity: int = 0) \
        -> Union[DiGraph, MultiDiGraph, Tuple[MultiDiGraph, DiGraph]]:
    """Return a NetworkX digraph from a pandas dataframe of a db dump

    Parameters
    ----------
    df : Union[str, pd.DataFrame]
        A dataframe, either as a file path to a file (.pkl or .csv) or a
        pandas DataFrame object.
    date : str
        A date string specifying when the data was dumped from the database.
    mesh_id_dict : dict
        A dict object mapping statement hashes to all mesh ids sharing a 
        common PMID
    graph_type : str
        Return type for the returned graph. Currently supports:
            - 'digraph': DiGraph (Default)
            - 'multidigraph': MultiDiGraph
            - 'signed': Tuple[DiGraph, MultiDiGraph]
            - 'signed-expanded': Tuple[DiGraph, MultiDiGraph]
            - 'digraph-signed-types':  DiGraph
    include_entity_hierarchies : bool
        If True, add edges between nodes if they are related ontologically
        with stmt type 'fplx': e.g. BRCA1 is in the BRCA family, so an edge
        is added between the nodes BRCA and BRCA1. Default: True. Note that
        this option only is available for the options directed/unsigned graph
        and multidigraph.
    sign_dict : Dict[str, int]
        A dictionary mapping a Statement type to a sign to be used for the
        edge. By default only Activation and IncreaseAmount are added as
        positive edges and Inhibition and DecreaseAmount are added as
        negative edges, but a user can pass any other Statement types in a
        dictionary.
    stmt_types : List[str]
        A list of statement types to epxand out to other signs
    z_sc_path:
        If provided, must be or be path to a square dataframe with HGNC symbols
        as names on the axes and floats as entries
    verbosity: int
        Output various messages if > 0. For all messages, set to 4.

    Returns
    -------
    Union[DiGraph, MultiDiGraph, Tuple[DiGraph, MultiDiGraph]]
        The type is determined by the graph_type argument
    """
    graph_options = ('digraph', 'multidigraph', 'signed', 'signed-expanded',
                     'digraph-signed-types')
    if graph_type.lower() not in graph_options:
        raise ValueError(f'Graph type {graph_type} not supported. Can only '
                         f'chose between {graph_options}')
    sign_dict = sign_dict if sign_dict else default_sign_dict

    graph_type = graph_type.lower()
    date = date if date else datetime.now().strftime('%Y-%m-%d')

    if isinstance(df, str):
        sif_df = file_opener(df)
    else:
        sif_df = df

    if z_sc_path is not None:
        if isinstance(z_sc_path, str):
            if z_sc_path.endswith('h5'):
                logger.info(f'Loading z-scores from {z_sc_path}')
                z_sc_df = pd.read_hdf(z_sc_path)
            elif z_sc_path.endswith('pkl'):
                logger.info(f'Loading z-scores from {z_sc_path}')
                z_sc_df: pd.DataFrame = file_opener(z_sc_path)
            else:
                raise ValueError(f'Unrecognized file: {z_sc_path}')
        elif isinstance(z_sc_path, pd.DataFrame):
            z_sc_df = z_sc_path
        else:
            raise ValueError('Only file paths and data frames allowed as '
                             'arguments to z_sc_path')
    else:
        z_sc_df = None

    # If signed types: filter out rows that of unsigned types
    if graph_type == 'digraph-signed-types':
        sif_df = sif_df[sif_df.stmt_type.isin(sign_dict.keys())]

    sif_df = sif_dump_df_merger(sif_df, graph_type, sign_dict, stmt_types,
                                mesh_id_dict, verbosity=verbosity)

    # Map ns:id to node name
    logger.info('Creating dictionary mapping (ns,id) to node name')
    ns_id_name_tups = set(
        zip(sif_df.agA_ns, sif_df.agA_id, sif_df.agA_name)).union(
        set(zip(sif_df.agB_ns, sif_df.agB_id, sif_df.agB_name)))
    ns_id_to_nodename = {(ns, _id): name for ns, _id, name in ns_id_name_tups}

    # Map hashes to edge for non-signed graphs
    if graph_type in {'multidigraph', 'digraph', 'digraph-signed-types'}:
        logger.info('Creating dictionary mapping hashes to edges for '
                    'unsigned graph')
        hash_edge_dict = {h: (a, b) for a, b, h in
                          zip(sif_df.agA_name,
                              sif_df.agB_name,
                              sif_df.stmt_hash)}

    # Create graph from df
    if graph_type == 'multidigraph':
        indranet_graph = IndraNet.from_df(sif_df)
    elif graph_type in ('digraph', 'digraph-signed-types'):
        # Flatten
        indranet_graph = IndraNet.digraph_from_df(sif_df,
                                                  'complementary_belief',
                                                  _weight_mapping)
    elif graph_type in ('signed', 'signed-expanded'):
        signed_edge_graph: MultiDiGraph = IndraNet.signed_from_df(
            df=sif_df, flattening_method='complementary_belief',
            weight_mapping=_weight_mapping
        )
        signed_node_graph: DiGraph = signed_edges_to_signed_nodes(
            graph=signed_edge_graph, copy_edge_data=True
        )
        signed_edge_graph.graph['date'] = date
        signed_node_graph.graph['date'] = date
        signed_edge_graph.graph['node_by_ns_id'] = ns_id_to_nodename
        signed_node_graph.graph['node_by_ns_id'] = ns_id_to_nodename

        # Get hash to signed edge mapping
        logger.info('Creating dictionary mapping hashes to edges for '
                    'unsigned graph')
        seg_hash_edge_dict = {} if graph_type == 'signed' else defaultdict(set)
        for edge in signed_edge_graph.edges:
            for es in signed_edge_graph.edges[edge]['statements']:
                if graph_type == 'signed':
                    seg_hash_edge_dict[es['stmt_hash']] = edge
                else:
                    seg_hash_edge_dict[es['stmt_hash']].add(edge)
        signed_edge_graph.graph['edge_by_hash'] = seg_hash_edge_dict

        sng_hash_edge_dict = {} if graph_type == 'signed' else defaultdict(set)
        for edge in signed_node_graph.edges:
            for es in signed_node_graph.edges[edge]['statements']:
                if graph_type == 'signed':
                    sng_hash_edge_dict[es['stmt_hash']] = edge
                else:
                    sng_hash_edge_dict[es['stmt_hash']].add(edge)
        signed_node_graph.graph['edge_by_hash'] = sng_hash_edge_dict
        if z_sc_df is not None:
            # Set z-score attributes
            add_corr_to_edges(graph=signed_edge_graph, z_corr=z_sc_df)
            add_corr_to_edges(graph=signed_node_graph, z_corr=z_sc_df)

        return signed_edge_graph, signed_node_graph
    else:
        raise ValueError(f'Unrecognized graph type {graph_type}. Must be one '
                         f'of: {", ".join(graph_options)}')

    if z_sc_df is not None:
        # Set z-score attributes
        add_corr_to_edges(graph=indranet_graph, z_corr=z_sc_df)

    # Add hierarchy relations to graph (not applicable for signed graphs)
    if include_entity_hierarchies and graph_type in ('multidigraph',
                                                     'digraph'):
        from depmap_analysis.network_functions.famplex_functions import \
            get_all_entities
        logger.info('Fetching entity hierarchy relationships')
        full_entity_list = get_all_entities()
        logger.info('Adding entity hierarchy manager as graph attribute')
        node_by_uri = {uri: _id for (ns, _id, uri) in full_entity_list}
        added_pairs = set()  # Save (A, B, URI)
        logger.info('Building entity relations to be added to data frame')
        entities = 0
        non_corr_weight = None
        if z_sc_df is not None:
            # Get non-corr weight
            for edge in indranet_graph.edges:
                if indranet_graph.edges[edge]['z_score'] == 0:
                    non_corr_weight = indranet_graph.edges[edge]['corr_weight']
                    break
            assert non_corr_weight is not None
            z_sc_attrs = {'z_score': 0, 'corr_weight': non_corr_weight}
        else:
            z_sc_attrs = {}

        for ns, _id, uri in full_entity_list:
            node = _id
            # Get name in case it's different than id
            if ns_id_to_nodename.get((ns, _id), None):
                node = ns_id_to_nodename[(ns, _id)]
            else:
                ns_id_to_nodename[(ns, _id)] = node

            # Add famplex edge
            for pns, pid in bio_ontology.get_parents(ns, _id):
                puri = get_identifiers_url(pns, pid)
                pnode = pid
                if ns_id_to_nodename.get((pns, pid), None):
                    pnode = ns_id_to_nodename[(pns, pid)]
                else:
                    ns_id_to_nodename[(pns, pid)] = pnode
                # Check if edge already exists
                if (node, pnode, puri) not in added_pairs:
                    entities += 1
                    # Belief and evidence are conditional
                    added_pairs.add((node, pnode, puri))  # A, B, uri of B
                    ed = {'agA_name': node, 'agA_ns': ns, 'agA_id': _id,
                          'agB_name': pnode, 'agB_ns': pns, 'agB_id': pid,
                          'stmt_type': 'fplx', 'evidence_count': 1,
                          'source_counts': {'fplx': 1}, 'stmt_hash': puri,
                          'belief': 1.0, 'weight': MIN_WEIGHT,
                          'curated': True,
                          'english': f'{pns}:{pid} is an ontological parent '
                                     f'of {ns}:{_id}',
                          'z_score': 0, 'corr_weight': 1}
                    # Add non-existing nodes
                    if ed['agA_name'] not in indranet_graph.nodes:
                        indranet_graph.add_node(ed['agA_name'],
                                                ns=ed['agA_ns'],
                                                id=ed['agA_id'])
                    if ed['agB_name'] not in indranet_graph.nodes:
                        indranet_graph.add_node(ed['agB_name'],
                                                ns=ed['agB_ns'],
                                                id=ed['agB_id'])
                    # Add edges
                    ed.pop('agA_id')
                    ed.pop('agA_ns')
                    ed.pop('agB_id')
                    ed.pop('agB_ns')
                    if indranet_graph.is_multigraph():
                        # MultiDiGraph
                        indranet_graph.add_edge(ed['agA_name'],
                                                ed['agB_name'],
                                                **ed)
                    else:
                        # DiGraph
                        u = ed.pop('agA_name')
                        v = ed.pop('agB_name')

                        # Check edge
                        if indranet_graph.has_edge(u, v):
                            indranet_graph.edges[(u, v)]['statements'].append(
                                ed)
                        else:
                            indranet_graph.add_edge(u,
                                                    v,
                                                    belief=1.0,
                                                    weight=1.0,
                                                    statements=[ed],
                                                    **z_sc_attrs)

        logger.info('Loaded %d entity relations into dataframe' % entities)
        indranet_graph.graph['node_by_uri'] = node_by_uri
    indranet_graph.graph['node_by_ns_id'] = ns_id_to_nodename
    indranet_graph.graph['edge_by_hash'] = hash_edge_dict
    indranet_graph.graph['date'] = date
    return indranet_graph


def _custom_pb_assembly(stmts_list=None):
    if stmts_list is None:
        logger.info('No statements provided, downloading latest pa stmt dump')
        stmts_list = get_latest_pa_stmt_dump()

    # Filter bad statements
    logger.info('Filtering out statements with bad position attribute')
    filtered_stmts = []
    for st in stmts_list:
        try:
            pos = getattr(st, 'position')
            try:
                if pos is not None and str(int(float(pos))) != pos:
                    continue
                else:
                    filtered_stmts.append(st)
            # Pos is not convertible to float
            except ValueError:
                continue
        # Not a statement with a position attribute
        except AttributeError:
            filtered_stmts.append(st)

    # Assemble Pybel model
    logger.info('Assembling PyBEL model')
    pb = PybelAssembler(stmts=filtered_stmts)
    pb_model = pb.make_model()
    return pb_model


def db_dump_to_pybel_sg(stmts_list=None, pybel_model=None, belief_dump=None,
                        default_belief=0.1, sign_in_edges=False):
    """Create a signed pybel graph from an evidenceless dump from the db

    Parameters
    ----------
    stmts_list : list[indra.statements.Statement]
        Provide a list of statements if they are already loaded. By default
        the latest available pa statements dump is downloaded from s3.
        Default: None.
    pybel_model : pybel.BELGraph
        If provided, skip generating a new pybel model from scratch
    belief_dump : dict
        If provided, reset the belief scores associated with the statements
        supporting the edges.
    default_belief : float
        Only used if belief_dump is provided. When no belief score is
        available, reset to this belief score. Default: 0.1.
    sign_in_edges : bool
        If True, check that all edges are stored with an index corresponding
        to the sign of the edge. Default: False.

    Returns
    -------
    tuple(DiGraph, MultiDiGraph)
    """
    # Get statement dump:
    # Look for latest file on S3 and pickle.loads it
    if pybel_model is None:
        pb_model = _custom_pb_assembly(stmts_list)
    else:
        logger.info('Pybel model provided')
        pb_model = pybel_model

    # If belief dump is provided, reset beliefs to the entries in it
    if belief_dump:
        logger.info('Belief dump provided, resetting belief scores')
        missing_hash = 0
        changed_belief = 0
        no_hash = 0
        logger.info(f'Looking for belief scores among {len(pb_model.edges)} '
                    f'edges')
        for edge in pb_model.edges:
            ed = pb_model.edges[edge]
            if ed and ed.get('stmt_hash'):
                h = ed['stmt_hash']
                if h in belief_dump:
                    ed['belief'] = belief_dump[h]
                    changed_belief += 1
                else:
                    logger.warning(f'No belief found for {h}')
                    ed['belief'] = default_belief
                    missing_hash += 1
            else:
                no_hash += 1
        logger.info(f'{no_hash} edges did not have hashes')
        logger.info(f'{changed_belief} belief scores were changed')
        logger.info(f'{missing_hash} edges did not have a belief entry')

    # Get a signed edge graph
    logger.info('Getting a PyBEL signed edge graph')
    pb_signed_edge_graph = belgraph_to_signed_graph(
        pb_model, symmetric_variant_links=True, symmetric_component_links=True,
        propagate_annotations=True
    )

    if sign_in_edges:
        for u, v, ix in pb_signed_edge_graph.edges:
            ed = pb_signed_edge_graph.edges[(u, v, ix)]
            if 'sign' in ed and ix != ed['sign']:
                pb_signed_edge_graph.add_edge(u, v, ed['sign'], **ed)
                pb_signed_edge_graph.remove_edge(u, v, ix)

    # Map hashes to edges
    logger.info('Getting hash to signed edge mapping')
    seg_hash_edge_dict = {}
    for edge in pb_signed_edge_graph.edges:
        if pb_signed_edge_graph.edges[edge].get('stmt_hash'):
            seg_hash_edge_dict[
                pb_signed_edge_graph.edges[edge]['stmt_hash']] = edge
    pb_signed_edge_graph.graph['edge_by_hash'] = seg_hash_edge_dict

    # Get the signed node graph
    logger.info('Getting a signed node graph from signed edge graph')
    pb_signed_node_graph = signed_edges_to_signed_nodes(
        pb_signed_edge_graph, copy_edge_data=True)

    # Map hashes to edges for signed nodes
    logger.info('Getting hash to edge mapping')
    sng_hash_edge_dict = {}
    for edge in pb_signed_node_graph.edges:
        if pb_signed_node_graph.edges[edge].get('stmt_hash'):
            sng_hash_edge_dict[
                pb_signed_node_graph.edges[edge]['stmt_hash']] = edge
    pb_signed_node_graph.graph['edge_by_hash'] = sng_hash_edge_dict

    logger.info('Done assembling signed edge and signed node PyBEL graphs')
    return pb_signed_edge_graph, pb_signed_node_graph


def rank_nodes(node_list, nested_dict_stmts, gene_a, gene_b, x_type):
    """Returns a list of tuples of nodes and their rank score

    The provided node list should contain the set of nodes that connects subj
    and obj through an intermediate node found in nested_dict_stmts.

    nested_dict_stmts

        d[subj][obj] = [stmts/stmt hashes]

    node_list : list[nodes]
    nested_dict_stmts : defaultdict(dict)
        Nested dict of statements: nest_d[subj][obj]
    gene_a : str
        Name of node A in an A-X-B connection
    gene_b : str
        Name of node B in an A-X-B connection
    x_type : str
        One of 'x_is_intermediary', 'x_is_downstream' or 'x_is_upstream'

    -------
    Returns
    dir_path_nodes_wb : list[(node, rank)]
        A list of node, rank tuples.
    """
    def _tuple_rank(ax_stmts, xb_stmts):
        def _body(t):
            assert len(t) == 2 or len(t) == 3
            bel = MIN_BELIEF
            if len(t) == 2:
                tp, hs = t
            elif len(t) == 3:
                tp, hs, bel = t
            else:
                raise IndexError('Tuple must have len(t) == 2,3 Tuple: %s' %
                                 repr(t))
            return tp, hs, bel
        ax_score_list = []
        xb_score_list = []
        for tup in ax_stmts:
            typ, hsh_a, belief = _body(tup)
            ax_score_list.append(belief)
        for tup in xb_stmts:
            typ, hsh_b, belief = _body(tup)
            xb_score_list.append(belief)
        return ax_score_list, xb_score_list

    def _dict_rank(ax_stmts, xb_stmts):
        ax_score_list = []
        xb_score_list = []
        for sd in ax_stmts:
            ax_score_list.append(float(sd.get('belief', MIN_BELIEF)))
        for sd in xb_stmts:
            xb_score_list.append(float(sd.get('belief', MIN_BELIEF)))
        return ax_score_list, xb_score_list

    def _calc_rank(nest_dict_stmts, subj_ax, obj_ax, subj_xb, obj_xb):
        ax_stmts = nest_dict_stmts[subj_ax][obj_ax]
        xb_stmts = nest_dict_stmts[subj_xb][obj_xb]
        hsh_a, hsh_b = None, None

        # The statment with the highest belief score should
        # represent the edge (potentially multiple stmts per edge)

        if isinstance(ax_stmts[0], tuple):
            ax_score_list, xb_score_list = _tuple_rank(ax_stmts, xb_stmts)
        elif isinstance(ax_stmts[0], (dict, defaultdict)):
            ax_score_list, xb_score_list = _dict_rank(ax_stmts, xb_stmts)

        # Rank by multiplying the best two belief scores for each edge
        rank = max(ax_score_list) * max(xb_score_list)

        # No belief score should be zero, thus rank should never be zero
        try:
            assert rank != 0
        except AssertionError:
            logger.warning('Combined rank == 0 for hashes %s and %s, implying '
                           'belief score is 0 for at least one of the '
                           'following statements: ' % (hsh_a, hsh_b))
        return rank

    dir_path_nodes_wb = []

    if x_type == 'x_is_intermediary':  # A->X->B or A<-X<-B
        for gene_x in node_list:
            x_rank = _calc_rank(nest_dict_stmts=nested_dict_stmts,
                                subj_ax=gene_a, obj_ax=gene_x,
                                subj_xb=gene_x, obj_xb=gene_b)
            dir_path_nodes_wb.append((gene_x, x_rank))

    elif x_type == 'x_is_downstream':  # A->X<-B
        for gene_x in node_list:
            x_rank = _calc_rank(nest_dict_stmts=nested_dict_stmts,
                                subj_ax=gene_a, obj_ax=gene_x,
                                subj_xb=gene_b, obj_xb=gene_x)
            dir_path_nodes_wb.append((gene_x, x_rank))
    elif x_type == 'x_is_upstream':  # A<-X->B

        for gene_x in node_list:
            x_rank = _calc_rank(nest_dict_stmts=nested_dict_stmts,
                                subj_ax=gene_x, obj_ax=gene_a,
                                subj_xb=gene_x, obj_xb=gene_b)
            dir_path_nodes_wb.append((gene_x, x_rank))

    return dir_path_nodes_wb


def ag_belief_score(belief_list):
    """Each item in `belief_list` should be a float"""
    # Aggregate belief score: 1-prod(1-belief_i)
    with np.errstate(all='raise'):
        try:
            ag_belief = np.longfloat(1.0) - np.prod(np.fromiter(map(
                lambda belief: np.longfloat(1.0) - belief, belief_list),
                dtype=np.longfloat)
            )
        except FloatingPointError as err:
            logger.warning('%s: Resetting ag_belief to 10*np.longfloat '
                           'precision (%.0e)' %
                           (err, Decimal(MIN_WEIGHT)))
            ag_belief = MIN_WEIGHT

    return ag_belief


def gilda_normalization(name: str, gilda_retry: bool = False) -> \
        Tuple[Union[None, str], Union[None, str], Union[None, str]]:
    """Query the grounding service for the most likely ns, id, name tuple

    Parameters
    ----------
    name: str
        Search gilda with this string
    gilda_retry: bool
        If True, try to reach gilda again after a previous perceived outage

    Returns
    -------
    Tuple[str, str, str]
        A 3-tuple containing the namespace, id, and normalized name of the
        search
    """
    global GILDA_TIMEOUT
    if gilda_retry and GILDA_TIMEOUT and gilda_pinger():
        logger.info('GILDA is responding again!')
        GILDA_TIMEOUT = False

    if GRND_URI and not GILDA_TIMEOUT:
        try:
            res = requests.post(GRND_URI, json={'text': name})
            if res.status_code == 200:
                rj = res.json()[0]['term']
                return rj['db'], rj['id'], rj['entry_name']
            else:
                logger.warning('Grounding service responded with code %d, '
                               'check your query format and URL' %
                               res.status_code)
        except IndexError:
            logger.info('No grounding exists for %s' % name)
        except ConnectionError:
            logger.warning('GILDA has timed out, ignoring future requests')
            GILDA_TIMEOUT = True
    else:
        if GILDA_TIMEOUT:
            logger.warning('Indra Grounding service not available.')
        else:
            logger.warning('Indra Grounding service not available. Add '
                           'GILDA_URL to `indra/config.ini`')
    return None, None, None


def pybel_node_name_mapping(pb_model, node_names=None, node_ns='HGNC'):
    """Generate a mapping of HGNC symbols to pybel nodes

    Parameters
    ----------
    node_names : iterable[str]
        Optional. An iterable containing the node names to be mapped to pybel
        nodes. If not provided, all nodes from the provided name space will be
        added.
    pb_model : PyBEL.Model
        An assembled pybel model
    node_ns : str
        The node namespace to consider. Default: HGNC.

    Returns
    -------
    dict
        A dictionary mapping names (HGNC symbols) to sets of pybel nodes
    """

    # Get existing node mappings
    corr_names = set(node_names) if node_names else set()
    pb_model_mapping = {}
    for node in pb_model.nodes:
        try:
            # Only consider HGNC nodes and if node name is in provided set
            # of HGNC symbol names
            if node.namespace == node_ns and \
                    ((corr_names and node.name in corr_names) or
                     not corr_names):
                if pb_model_mapping.get(node.name):
                    pb_model_mapping[node.name].add(node)
                else:
                    pb_model_mapping[node.name] = {node}
            else:
                continue
        # No attribute 'name' or 'namespace'
        except AttributeError:
            continue
    return pb_model_mapping


def yield_multiple_paths(g, sources, path_len=None, **kwargs):
    """Wraps bfs_search and cycles between one generator per source in sources

    Parameters
    ----------
    g : DiGraph
    sources : list
    path_len : int
        Only produce paths of this length (number of edges)
    kwargs : **kwargs
    """
    # create one generator per drug
    generators = []
    cycler = cycle(range(len(sources)))
    for source in sources:
        generators.append(bfs_search(g, source, **kwargs))

    skip = set()
    while True:
        gi = next(cycler)
        if len(skip) >= len(sources):
            break
        # If gi in skip, get new one, unless we added all of them
        while gi in skip and len(skip) < len(sources):
            gi = next(cycler)
        try:
            path = next(generators[gi])
            if path_len:
                if path_len > len(path):
                    # Too short
                    continue
                elif path_len == len(path):
                    yield path
                elif path_len < len(path):
                    # Too long: Done. Add to skip.
                    skip.add(gi)
                    continue
            # No path length specified, yield all
            else:
                yield path
        except StopIteration:
            print(f'Got StopIteration from {gi}')
            skip.add(gi)
