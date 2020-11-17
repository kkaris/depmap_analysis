"""Explainer and helper functions for depmap_script2.py When adding new
explanation functions, please also add them to the mapping at the end"""
import logging
from functools import wraps
from pybel.dsl import BaseEntity, ComplexAbundance, Reaction

import pandas as pd
from networkx import DiGraph, MultiDiGraph
from typing import Tuple, Dict, List, ClassVar, Union, Iterable, Optional, Set

from indra.ontology.standardize import standardize_agent_name
from indra.explanation.model_checker import PybelModelChecker
from indra.explanation.pathfinding import bfs_search_multiple_nodes
from indra.statements import Statement, RegulateAmount, \
    RegulateActivity, get_all_descendants, Agent, AddModification,\
    RemoveModification
from emmaa.queries import OpenSearchQuery
from depmap_analysis.network_functions.famplex_functions import common_parent
from depmap_analysis.network_functions.net_functions import gilda_normalization, \
    INT_PLUS, INT_MINUS


__all__ = ['explained', 'expl_ab', 'expl_ba', 'expl_axb', 'expl_bxa',
           'find_cp', 'get_sd', 'get_sr', 'get_st',
           'get_ns_id_name_pybel_node', 'get_ns_id',
           'normalize_corr_names', 'expl_functions', 'funcname_to_colname']
graph_types = {'unsigned', 'signed', 'pybel'}


def _issubclass_excl(Cls: Statement, ClsOther: Union[Iterable, Statement]) \
        -> bool:
    try:
        return any(_issubclass_excl(Cls, CO) for CO in ClsOther)
    except TypeError:
        return issubclass(Cls, ClsOther) and Cls.__name__ != ClsOther.__name__


_allowed_stmt_types = (AddModification, RemoveModification, RegulateAmount,
                       RegulateActivity)
pb_stmt_classes = [StmtCls for StmtCls in get_all_descendants(Statement)
                   if _issubclass_excl(StmtCls, _allowed_stmt_types)]
pb_stmt_names = [cls.__name__ for cls in pb_stmt_classes]


logger = logging.getLogger(__name__)


def expl_func(ef):
    @wraps(ef)
    def decorator(*args, **kwargs):
        _type = args[4]
        if _type not in graph_types:
            raise ValueError(f'Unknown graph type {_type}. Graph type must '
                             f'be one of {", ".join(graph_types)}')
        return ef(*args, **kwargs)
    return decorator


def explained(s, o, corr, net, _type, **kwargs):
    # This function is used for a priori explained relationships
    return s, o, 'explained_set'


def find_cp(s, o, corr, net, _type, **kwargs):
    if _type == 'pybel':
        s_name = kwargs['s_name']
        s_ns, s_id = get_ns_id_pybel_node(hgnc_sym=s_name, node=s)
        o_name = kwargs['o_name']
        o_ns, o_id = get_ns_id_pybel_node(hgnc_sym=o_name, node=o)
    else:
        s_name = s
        o_name = o
        s_ns, s_id, o_ns, o_id = get_ns_id(s, o, net)

    if not s_id:
        s_ns, s_id, s_norm_name = gilda_normalization(s_name)
    if not o_id:
        o_ns, o_id, o_norm_name = gilda_normalization(o_name)

    if s_id and o_id:
        # Possible kwargs:
        #   - immediate_only : bool
        #         Determines if all or just the immediate parents should be
        #         returned. Default: False.
        #   - is_a_part_of : iterable
        #         If provided, the parents must be in this set of ids. The
        #         set is assumed to be valid ontology labels (see
        #         ontology.label()).
        parents = list(common_parent(
            ns1=s_ns, id1=s_id, ns2=o_ns, id2=o_id,
            immediate_only=kwargs.get('immediate_only', False),
            is_a_part_of=kwargs.get('is_a_part_of')
        ))
        if parents:
            # if kwargs.get('ns_set'):
            #     parents = {(ns, _id) for ns, _id in parents if ns.lower() in
            #                kwargs['ns_set']} or None
            return s, o, parents

    return s, o, None


def expl_axb(s, o, corr, net, _type, **kwargs):
    if _type in {'signed', 'unsigned'}:
        x_set = set(net.succ[s]) & set(net.pred[o])
        x_nodes = _get_signed_interm(s, o, corr, net, x_set) \
            if _type == 'signed' else x_set
    elif _type == 'pybel':
        # Here, x_nodes is a tuple of (one_edge_paths, two_edge_paths)
        x_nodes = get_pb_paths(pbmc=net, max_path_len=3, **kwargs)
    else:
        raise ValueError(f'Unhandled type {_type}')

    # Filter ns
    if kwargs.get('ns_set'):
        if _type == 'pybel':
            # Only check the two edge paths
            oe, te = x_nodes
            te = [p for p in te if get_ns_id_name_pybel_node(p[1])[0].lower()
                  in kwargs['ns_set']]
            x_nodes = oe, te
        else:
            x_nodes = {x for x in x_nodes if net.nodes[x]['ns'].lower() in
                       kwargs['ns_set']} or None

    if x_nodes:
        return s, o, list(x_nodes)
    else:
        return s, o, None


def expl_bxa(s, o, corr, net, _type, **kwargs):
    if _type == 'pybel':
        options = _flip_options(**kwargs)
    else:
        options = kwargs
    return expl_axb(o, s, corr, net, _type, **options)


# Shared regulator: A<-X->B
def get_sr(s, o, corr, net, _type, **kwargs):
    if _type in {'signed', 'unsigned'}:
        x_set = set(net.pred[s]) & set(net.pred[o])
        x_nodes = _get_signed_interm(s, o, corr, net, x_set) \
            if _type == 'signed' else x_set
    elif _type == 'pybel':
        x_nodes = get_shared_interactors_pb(pbmc=net, corr=corr,
                                            reverse=True, **kwargs)
    else:
        raise ValueError(f'Unhandled type {_type}')

    # Filter ns
    if kwargs.get('ns_set') and _type != 'pybel':
        x_nodes = {x for x in x_nodes if
                   net.nodes[x]['ns'].lower() in kwargs['ns_set']} or None

    if x_nodes:
        return s, o, list(x_nodes)
    else:
        return s, o, None


# Shared target: A->X<-B
def get_st(s, o, corr, net, _type, **kwargs):
    if _type in {'signed', 'unsigned'}:
        x_set = set(net.succ[s]) & set(net.succ[o])
        x_nodes = _get_signed_interm(s, o, corr, net, x_set) \
            if _type == 'signed' else x_set
    elif _type == 'pybel':
        x_nodes = get_shared_interactors_pb(pbmc=net, corr=corr,
                                            reverse=False, **kwargs)
    else:
        raise ValueError(f'Unhandled type {_type}')

    # Filter ns
    if kwargs.get('ns_set') and _type != 'pybel':
        x_nodes = {x for x in x_nodes if
                   net.nodes[x]['ns'].lower() in kwargs['ns_set']} or None

    if x_nodes:
        return s, o, list(x_nodes)
    else:
        return s, o, None


def get_sd(s, o, corr, net, _type, **kwargs):
    def get_nnn_set(n: str, g: MultiDiGraph, signed: bool) \
            -> Set[Union[str, Tuple[str, str]]]:
        n_x_set = set()
        for x in g.succ[n]:
            # If signed, add edges instead and match sign in helper
            if signed:
                for y in g.succ[x]:
                    n_x_set.add((x, y))
            # Just add nodes for unsigned
            else:
                n_x_set.add(x)
                n_x_set.update(g.succ[x])
        return n_x_set

    # Get next-nearest-neighborhood for subject
    s_x_set = get_nnn_set(s, net, _type in {'signed', 'pybel'})
    o_x_set = get_nnn_set(o, net, _type in {'signed', 'pybel'})

    # Get intersection of each nodes' 1st & 2nd layer neighbors
    x_set = s_x_set & o_x_set
    x_set_union = s_x_set | o_x_set

    if _type in {'signed', 'pybel'}:
        x_nodes = _get_signed_deep_interm(s, o, corr, net, x_set, False)
        x_nodes_union = _get_signed_deep_interm(s, o, corr, net, x_set_union,
                                                True)
    else:
        x_nodes = x_set
        x_nodes_union = x_set_union

    # Filter ns
    if kwargs.get('ns_set'):
        x_nodes = {x for x in x_nodes if net.nodes[x]['ns'].lower() in
                   kwargs['ns_set']} or None
        x_nodes_union = {x for x in x_nodes_union if net.nodes[x][
            'ns'].lower() in kwargs['ns_set']} or None

    if x_nodes:
        return s, o, (list(x_nodes), list(x_nodes_union))
    else:
        return s, o, None


def expl_ab(s, o, corr, net, _type, **kwargs):
    edge_dict = get_edge_statements(s, o, corr, net, _type, **kwargs)
    if edge_dict:
        return s, o, edge_dict.get('stmt_hash') if _type == 'pybel' else \
            edge_dict.get('statements')
    return s, o, None


def expl_ba(s, o, corr, net, _type, **kwargs):
    if _type == 'pybel':
        options = _flip_options(**kwargs)
    else:
        options = kwargs
    return expl_ab(o, s, corr, net, _type, **options)


def _flip_options(**kwargs):
    options = {
        'o_name': kwargs.pop('s_name'),
        'o_ns': kwargs.pop('s_ns'),
        'o_id': kwargs.pop('s_id'),
        's_name': kwargs.pop('o_name'),
        's_ns': kwargs.pop('o_ns'),
        's_id': kwargs.pop('o_id')
    }
    for k, v in kwargs.items():
        if k not in options:
            options[k] = v
    return options


def get_edge_statements(s, o, corr, net, _type, **kwargs):
    if _type in {'signed', 'pybel'}:
        int_sign = INT_PLUS if corr >= 0 else INT_MINUS
        return net.edges.get((s, o, int_sign), None)
    else:
        return net.edges.get((s, o))


def _get_signed_interm(s, o, corr, sign_edge_net, x_set):
    # Make sure we have the right sign type
    int_sign = INT_PLUS if corr >= 0 else INT_MINUS

    # ax and xb sign need to match correlation sign
    x_approved = set()
    for x in x_set:
        ax_plus = (s, x, INT_PLUS) in sign_edge_net.edges
        ax_minus = (s, x, INT_MINUS) in sign_edge_net.edges
        xb_plus = (x, o, INT_PLUS) in sign_edge_net.edges
        xb_minus = (x, o, INT_MINUS) in sign_edge_net.edges

        if int_sign == INT_PLUS:
            if ax_plus and xb_plus or ax_minus and xb_minus:
                x_approved.add(x)
        if int_sign == INT_MINUS:
            if ax_plus and xb_minus or ax_minus and xb_plus:
                x_approved.add(x)
    return x_approved


def _get_signed_deep_interm(
        s: str, o: str, corr: float, sign_edge_net: MultiDiGraph,
        xy_set: Set[Tuple[str, str]], union: bool) -> Set[str]:
    # Make sure we have the right sign type
    path_sign = INT_PLUS if corr >= 0 else INT_MINUS

    # a-x-y and b-x-y need to both match path sign
    x_approved = set()
    for x, y in xy_set:
        sx_plus = (s, x, INT_PLUS) in sign_edge_net.edges
        ox_plus = (o, x, INT_PLUS) in sign_edge_net.edges
        xy_plus = (x, y, INT_PLUS) in sign_edge_net.edges
        sx_minus = (s, x, INT_MINUS) in sign_edge_net.edges
        ox_minus = (o, x, INT_MINUS) in sign_edge_net.edges
        xy_minus = (x, y, INT_MINUS) in sign_edge_net.edges

        # Match args for _approve_signed_paths
        args = (sx_minus, sx_plus, ox_minus, ox_plus, xy_minus, xy_plus,
                path_sign, union)

        # Add nodes that form paths with the correct sign
        if _approve_signed_paths(*args):
            x_approved.update({x, y})
    return x_approved


def _approve_signed_paths(sxm: bool, sxp: bool, oxm: bool, oxp: bool,
                          xym: bool, xyp: bool, sign: int, union: bool) \
        -> bool:
    def _asp(n1: bool, n2: bool, p1: bool, p2: bool, s: int) -> bool:
        # Approve Signed Path
        if s == INT_PLUS:
            return p1 and p2 or n1 and n2
        else:
            return p1 and n2 or n1 and p2

    # Match args for _asp
    sargs = (sxm, xym, sxp, xyp, sign)
    oargs = (oxm, xym, oxp, xyp, sign)
    if union:
        return _asp(*sargs) or _asp(*oargs)
    else:
        return _asp(*sargs) and _asp(*oargs)


def get_ns_id(subj, obj, net):
    """Get ns:id for both subj and obj

    Note: should *NOT* be used with PyBEL nodes

    Parameters
    ----------

    subj : str
        The subject node
    obj : str
        The source node
    net : Graph
        A networkx graph object that at least contains node entries.

    Returns
    -------
    tuple
        A tuple with four entries:
        (subj namespace, subj id, obj namespace, obj id)
    """
    s_ns = net.nodes[subj]['ns'] if net.nodes.get(subj) else None
    s_id = net.nodes[subj]['id'] if net.nodes.get(subj) else None
    o_ns = net.nodes[obj]['ns'] if net.nodes.get(obj) else None
    o_id = net.nodes[obj]['id'] if net.nodes.get(obj) else None
    return s_ns, s_id, o_ns, o_id


def get_ns_id_name_pybel_node(node: Union[BaseEntity, Tuple[BaseEntity]],
                              hgnc_sym: str = None) -> \
        Tuple[Union[None, str], Union[None, str], Union[None, str]]:
    """

    Parameters
    ----------
    hgnc_sym : str
        Name to match
    node : Optional[Union[BaseEntity, Tuple]]
        PyBEL node or tuple of PyBEL nodes

    Returns
    -------
    tuple
        Tuple of ns, id, name for node
    """
    # If tuple of nodes, recursive call until match is found
    if isinstance(node, tuple):
        for n in node:
            ns, _id, name = get_ns_id_name_pybel_node(n, hgnc_sym)
            if ns is not None:
                return ns, _id, name
        logger.warning('None of the names in the tuple matched the HGNC '
                       'symbol')
        return None, None, None
    # If PyBEL node, check name match, return if match, else None tuple
    elif isinstance(node, BaseEntity):
        try:
            ns, _id, name = _get_pb_ns_id(node)
            if hgnc_sym and _id == hgnc_sym:
                return ns, _id, name
            elif hgnc_sym is None and ns is not None:
                return ns, _id, name
            else:
                return None, None, None
        except AttributeError:
            return None, None, None
    # Not recognized
    else:
        logger.warning(f'Type {node.__class__} not recognized')
        return None, None, None


def _get_pb_ns_id(pbn: BaseEntity) -> Tuple[str, str, str]:
    if isinstance(pbn, ComplexAbundance):
        return _get_pb_ns_id(pbn.members[0])
    if isinstance(pbn, Reaction):
        return _get_pb_ns_id(pbn.products[0])
    return pbn.namespace, pbn.identifier, pbn.name


def get_pb_paths(s_name: str, s_ns: str, s_id: str, o_name: str, o_ns: str,
                 o_id: str, pbmc: PybelModelChecker, max_path_len: int,
                 pybel_stmt_types: List[Statement] = None) \
        -> Tuple[List[List[BaseEntity]], List[List[BaseEntity]]]:
    """Given HGNC symbols a & b, find paths using the provided model checker

    The paths are directed paths of a->b or a->x->b

    Parameters
    ----------
    s_name : str
    s_ns : str
    s_id : str
    o_name : str
    o_ns : str
    o_id : str
    pbmc : PybelModelChecker
        The PyBelModelChecker used to find paths between a and b
    pybel_stmt_types : List
    max_path_len : Optional[int]

    Returns
    -------

    """
    pybel_stmt_types = pb_stmt_classes if pybel_stmt_types is None else \
        pybel_stmt_types
    one_edge_results = set()
    two_edge_results = set()
    for StmtClass in pybel_stmt_types:
        stmt = StmtClass(Agent(s_name, db_refs={s_ns: s_id}),
                         Agent(o_name, db_refs={o_ns: o_id}))
        mc_res = pbmc.check_statement(stmt=stmt, max_paths=10000,
                                      max_path_length=max_path_len)
        if mc_res.path_found:
            for path in mc_res.paths:
                if len(path) == 2:
                    one_edge_results.add([p[0] for p in path])
                elif len(path) == 3:
                    two_edge_results.add([p[0] for p in path])

    return list(one_edge_results), list(two_edge_results)


def get_shared_interactors_pb(
        s_name: str, s_ns: str, s_id: str, o_name: str, o_ns: str, o_id: str,
        pbmc: PybelModelChecker, corr: float, reverse: bool,
        pybel_stmt_types: List[Statement] = None) \
        -> Set[Tuple[Tuple[BaseEntity, int]]]:
    """

    Parameters
    ----------
    s_name : str
    s_ns : str
    s_id : str
    o_name : str
    o_ns : str
    o_id : str
    pbmc : PybelModelChecker
    corr : float
    reverse : bool
        True if upstream, False if downstream
    pybel_stmt_types : List[Statement]

    Returns
    -------

    """
    pybel_stmt_types = pb_stmt_names if pybel_stmt_types is None else \
        pybel_stmt_types
    a_pos = set()
    a_neg = set()
    b_pos = set()
    b_neg = set()
    pos, neg = 0, 1
    corr_sign = pos if corr > 0 else neg
    for stmt_type in pybel_stmt_types:
        # Do query for a
        ag_a = Agent(s_name, db_refs={s_ns: s_id})
        standardize_agent_name(ag_a)
        query_a = OpenSearchQuery(ag_a, stmt_type, 'subject', ['HGNC', 'FPLX'])
        # Get both signs
        a_pos.update(_get_neigh(query_a, pbmc, reverse, sign=pos))
        a_neg.update(_get_neigh(query_a, pbmc, reverse, sign=neg))

        # Do query for b
        ag_b = Agent(o_name, db_refs={o_ns: o_id})
        standardize_agent_name(ag_b)
        query_b = OpenSearchQuery(ag_b, stmt_type, 'subject', ['HGNC', 'FPLX'])
        b_pos.update(_get_neigh(query_b, pbmc, reverse, sign=pos))
        b_neg.update(_get_neigh(query_b, pbmc, reverse, sign=neg))

        # todo: break loop early if we have results?

    # Match up results with sign
    nodes1, nodes2 = set(), set()
    if corr_sign == pos:
        if a_pos and b_pos or a_neg and b_neg:
            nodes1 = a_pos & b_pos
            nodes2 = a_neg & b_neg
    else:
        if a_pos and b_neg or a_neg and b_pos:
            nodes1 = a_pos & b_neg
            nodes2 = a_neg & b_pos
    shared_interactors = nodes1 | nodes2
    return shared_interactors


def _get_neigh(query: OpenSearchQuery, pbmc: PybelModelChecker, reverse: bool,
               sign: int) -> Set[Tuple[BaseEntity]]:
    """

    Parameters
    ----------
    query :  OpenSearchQuery
    pbmc : PybelModelChecker
    reverse : bool
        True if upstream, False if downstream
    sign : int
        The target sign of the search

    Returns
    -------
    List[Tuple[Tuple[BaseEntity, int]]]
    """
    # Do open search query
    subj_nodes, obj_nodes, res_code = pbmc.get_all_subjects_objects(
        stmt=query.path_stmt)
    nodes = obj_nodes if reverse else subj_nodes
    if nodes:
        path_gen = bfs_search_multiple_nodes(
            g=pbmc.get_graph(), source_nodes=nodes, reverse=reverse, sign=sign,
            depth_limit=1)
        return {tuple(n[0] for n in path) for path in path_gen}
    else:
        return set()


def normalize_corr_names(corr_m: pd.DataFrame,
                         graph: Union[DiGraph, MultiDiGraph],
                         ns: str = None) -> pd.DataFrame:
    # todo:
    #  1. Move this function, together with get_ns_id,
    #     get_ns_id_pybel_node, normalize_entitites to net_functions
    #  2. Provide ns and id to the correlation matrix here too (requires
    #     overhaul of depmap script)
    #  3. Add support for pybel
    """

    Parameters
    ----------
    corr_m : pd.DataFrame
        A square pandas dataframe representing a correlation matrix. It is
        assumed that columns and indices are identical.
    graph : Union[DiGraph, MultiDiGraph]
        A graph to look in to see if the names are there
    ns : str
        The assumed namespace of the names in corr_m

    Returns
    -------
    pd.DataFrame
    """
    def _get_ns_id(n: str, g: Union[DiGraph, MultiDiGraph]) -> Tuple[str, str]:
        return g.nodes[n]['ns'], g.nodes[n]['id']

    col_names = corr_m.columns.values
    normalized_names = []
    for name in col_names:
        if name in graph.nodes:
            normalized_names.append(name)
        else:
            ns, _id, nn = gilda_normalization(name)
            if nn:
                normalized_names.append(nn)
            else:
                normalized_names.append(name)

    # Reset the normalized names
    corr_m.columns = normalized_names
    corr_m.index = normalized_names

    return corr_m


# Add new function to the tuple
expl_func_list = (explained, expl_ab, expl_ba, expl_axb, expl_bxa, find_cp,
                  get_sr, get_st, get_sd)
# Map the name of the function to a more human friendly column name
funcname_to_colname = {
    'explained': 'explained set',
    'expl_ab': 'a-b',
    'expl_ba': 'b-a',
    'expl_axb': 'a-x-b',
    'expl_bxa': 'b-x-a',
    'find_cp': 'common parent',
    'get_sr': 'shared regulator',
    'get_st': 'shared target',
    'get_sd': 'shared downstream'
}
expl_functions = {f.__name__: f for f in expl_func_list}
assert len(expl_func_list) == len(funcname_to_colname)
