from oeip.utils.trees import Tree
import networkx as nx
from quark import HardwareAdjacency, Embedding, PolyIsing, Objective # VariableMapping, Polynomial, PolyBinary
from oeip.get import get_optimal_embedded_ising_objective
import dimod
from typing import List, Dict, Tuple, Optional, Union
import dwave
from itertools import combinations, product

SIGMA = 'sigma'
SUBTREE_SIGMA_SUM = 'subtree_sigma_sum'


def new_add_arcs(self, edges):
    """ 
    add the arcs to the digraph showing in a certain direction
    
    MODIFIED to find spaninng trees using nx.random_spanning_tree!
    """
    graph = nx.Graph()
    graph.add_edges_from(edges)
    if not nx.is_tree(graph):
        tree = nx.random_spanning_tree(graph)
        edges = tree.edges

    self.add_edges_from(edges)
    assert nx.is_tree(self.to_undirected()), 'The edges need to form a tree'

    # the edges will be transformed to arcs always showing to the subtree with the lower sigma sum value
    for arc in edges:
        _, nodes = self.get_cut_node_sets(arc)
        sigma_sum_in, sigma_sum_out = self.get_sigma_sums(nodes)
        self.arcs[arc][SUBTREE_SIGMA_SUM] = sigma_sum_in
        if sigma_sum_out < sigma_sum_in:
            self.remove_edge(*arc)
        if sigma_sum_out <= sigma_sum_in:
            # if the sigma sums are equal, we have arcs showing in both directions
            reverse_arc = arc[::-1]
            self.add_edge(*reverse_arc)
            self.arcs[reverse_arc][SUBTREE_SIGMA_SUM] = sigma_sum_out

## updated CLASS method!
Tree._add_arcs = new_add_arcs


def opt_embedding_quark_ising(
                  embedded_problem: Dict[int, List[int]],
                  hardware_graph:   nx.Graph,
                  linear_terms:     Dict[int, float],
                  quadratic_terms:  Dict[Tuple[int, int], float],
                  constant_offset:Optional[float] = 0
                  ) -> Tuple[Objective, Objective]:
    """
    Given an embedded problem, choose chain strengths s.t. embedding 
    is correct (better than dwaves current chain strength choice!)

    Note embedded_problem must be defined for ISING problem, not QUBO!
    
    Returns:
        obj (Objective): PolyIsing problem function 
        opt_embedded_ising (Objective): embedded PolyIsing problem function
        opt_bqm_ising_emb (dimod.BinaryQuadraticModel): embedded dwave problem with correct chain strengths!

    Note each output object can be solved using ScipModel in quark!

        from quark import ScipModel
        model = ScipModel.get_from_objective(obj)
        solution = model.solve()
        soln_dict = dict(solution.items())
        print(solution.solving_status, solution.objective_value, soln_dict)

    ALSO BQM output can be converted to binary model by opt_bqm_ising_emb.to_qubo() (same with the polynomials
    in each of the Objective outputs: obj.polynomial.to_binary()  )

    """
    
    hwa = HardwareAdjacency(hardware_graph.edges, "hwa")
    # embedding_obj = Embedding.get_from_hwa(embedded_problem, hwa)
    # assert embedding_obj.is_valid()

    embedding_obj = manual_embedding_quark(embedded_problem, 'embedded', hwa)
    
    all_terms = {(key, ): val for key, val in linear_terms.items()}
    all_terms.update(quadratic_terms)
    if abs(constant_offset)>0:
        all_terms.update({(): constant_offset})    
    ising_poly = PolyIsing(all_terms)

    assert embedding_obj.is_valid(couplings=ising_poly.get_graph().edges,
                                  hwa=hwa), 'embedded problem not correct!'


    obj = Objective(ising_poly, name='objective_embedded') 
    opt_embedded_ising = get_optimal_embedded_ising_objective(obj,
                                                              embedding_obj)
    opt_bqm_ising_emb = dimod.BinaryQuadraticModel.from_ising({key[0]: val for key, val in opt_embedded_ising.polynomial.linear.items()},
                                                    dict(opt_embedded_ising.polynomial.quadratic.items()),
                                                    opt_embedded_ising.polynomial.offset)
        
    
    return obj, opt_embedded_ising, opt_bqm_ising_emb


### FASTER VERSION OF:
# from quark.embedding import get_var_edges_map, get_coupling_edges_map
def get_edges_map(var_nodes_map, hwa):
    """
    get the edges in the embedding subgraph for each variable and the edges connecting the embedding subgraphs of each variable pair

    :param (dict or list[list]) var_nodes_map: mapping of variables to nodes in the hardware graph
    :param (list[tuples] or HardwareAdjacency) hwa: hardware adjacency defining the hardware graph
    :return: a tuple containing the mapping of variables to edges in the hardware graph and the mapping of couplings to edges in the hardware graph
    """
    # pre-comute neighbored
    are_neighbored = _get_are_neighbored_func(hwa)

    var_edges_map = {var: list(get_edges_among(nodes, are_neighbored)) for var, nodes in var_nodes_map.items()}
    coupling_edges_map = {tuple(sorted((n1, n2))): sorted(list(get_edges_between(var_nodes_map[n1], 
                                                                          var_nodes_map[n2], are_neighbored)))
                                                                          for n1, n2 in combinations(var_nodes_map.keys(), 2)}
    return var_edges_map, coupling_edges_map


def get_edges_among(nodes, are_neighbored):
    """
    get the edges in the embedding subgraph defined by the given nodes

    :param (set or list) nodes: set of nodes in the hardware graph
    :param (function) are_neighbored: function to check if two nodes are neighbors
    :return: the corresponding edges in the hardware graph
    """
    nodes = list(nodes)
    return (tuple(sorted((n1, n2))) for n1, n2 in combinations(nodes, 2) if are_neighbored(n1, n2))


def get_edges_between(nodes1, nodes2, are_neighbored):
    """
    get the edges connecting the embedding subgraphs defined by the two node sets

    :param (set or list) nodes1: first set of nodes in the hardware graph
    :param (set or list) nodes2: second set of nodes in the hardware graph
    :param (function) are_neighbored: function to check if two nodes are neighbors
    :return: the edges in the hardware graph connecting the corresponding embedding subgraphs
    """
    nodes1, nodes2 = list(nodes1), list(nodes2)
    return (tuple(sorted((n1, n2))) for n1, n2 in product(nodes1, nodes2) if are_neighbored(n1, n2))


def _get_are_neighbored_func(hwa):
    hwa_set = set(hwa)
    return lambda node1, node2: (node1, node2) in hwa_set or (node2, node1) in hwa_set


def manual_embedding_quark(embedded_problem: Dict[int, List[int]], name:str, hwa):
    """
    Get embedding via manual approach
    """
    # hwa = HardwareAdjacency(hardware_graph.edges, "hwa")


    # from quark.embedding import get_var_edges_map, get_coupling_edges_map
    # var_edges_map = get_var_edges_map(embedded_problem, hwa)
    # coupling_edges_map = get_coupling_edges_map(embedded_problem, hwa)

    var_edges_map, coupling_edges_map = get_edges_map(embedded_problem, hwa)

    embedding = Embedding(var_nodes_map=embedded_problem,
                    coupling_edges_map=coupling_edges_map, 
                    var_edges_map=var_edges_map,
                    name=name)
    
    return embedding

