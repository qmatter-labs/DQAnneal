import dimod
from minorminer import find_embedding
from dwave import embedding
from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
from dwave import embedding

def get_problem_graph(dwave_qubo) -> Tuple[nx.Graph,
                                           Dict[int, List[int]]]:

    source_bqm_qubo = dimod.BinaryQuadraticModel.from_qubo(dwave_qubo)
    problem_graph = dimod.to_networkx_graph(source_bqm_qubo)

    node_edge_dict = {node: list(problem_graph.neighbors(node)) 
                      for node in problem_graph.nodes}

    return problem_graph, node_edge_dict


def embed_problem_onto_hardware(problem_graph: dimod.BinaryQuadraticModel,
                    hardware_graph: nx.Graph,
                    maxtime_sec:int=120,
                    attempts:int=10) -> Tuple[Dict[Tuple[int, int], float],
                                              int]:
    
    embedded_problem, valid_flag = find_embedding(problem_graph,
                                                list(hardware_graph.edges), 
                                                return_overlap=True, 
                                                timeout=maxtime_sec,
                                                tries= attempts)
    return embedded_problem, valid_flag


def define_embedded_qubo_problem(embedded_problem: Dict[Tuple[int, int], float],
                                 edge_dict_hardware:dict,
                                 qubo_Q: dict):
    """

    Args:
        embedded_problem (dict): embedded problem (obtained from embed_problem_onto_hardware function)
        edge_dict (dict): dictionary of edges for each node.
        qubo_Q (dict): Q dict of indices as keys and value as coefficient.
    
    """
    trans_qubo = embedding.embed_qubo(qubo_Q,
                                        embedded_problem, 
                                        edge_dict_hardware)
    

    # pubo_result = boolean assigment as dict
    # pubo_value  = value of pubo given the pubo result!
    convert_fn_sample = lambda pubo_result, pubo_value: dimod.SampleSet.from_samples([pubo_result],
                                                                dimod.vartypes.Vartype.BINARY,
                                                                energy=[pubo_value])

    return trans_qubo, convert_fn_sample


def define_embedded_ising_problem(linear_dict: Dict[int, float],
                                  quadratic_dict: Dict[Tuple[int, int], float],
                                  embedded_problem: Dict[Tuple[int, int], float],
                                  edge_dict_hardware:dict):
    """
    Args:
        linear_dict (dict): linear terms of ising problem
        quadratic_dict (dict): quadratic terms of ising problem.
        embedded_problem (dict): embedded problem (obtained from embed_problem_onto_hardware function)
        edge_dict (dict): dictionary of edges for each node.
    
    Returns:
        linear_dict_trans (dict): linear terms in ising problem.
        quadratic_dict_trans (dict): quadratic terms in ising problem.
        unique_idxs (array): list of unique node idx of problem (gives total number of spins!)
        convert_fn_sample (callable): lambda function that maps dimod result!
        
    """
    linear_dict_trans, quadratic_dict_trans = embedding.embed_ising(linear_dict, quadratic_dict,
                                            embedded_problem, 
                                            edge_dict_hardware)
    # ## merge dictionaries!
    # ising_trans_problem = {(key,): value for key, value in linear_dict_trans.items()} | quadratic_dict_trans

    unique_idxs = np.union1d(np.array(list(linear_dict_trans.keys())),
                  np.array(list(quadratic_dict_trans.keys())).flatten())

    convert_fn_sample = lambda puso_result, puso_value: dimod.SampleSet.from_samples([puso_result],
                                                                    dimod.vartypes.Vartype.SPIN,
                                                                    energy=[puso_value])
    
    return linear_dict_trans, quadratic_dict_trans, unique_idxs, convert_fn_sample


def unembed_samples(embedded_sampleset: dimod.sampleset.SampleSet,
                    embedded_problem: Dict,
                    source_bqm:dimod.BinaryQuadraticModel) -> dimod.sampleset.SampleSet:
    """
    get solution of to original problem from embedded solution!
    """
    converted_samples = embedding.unembed_sampleset(embedded_sampleset,
                                    embedded_problem, 
                                    source_bqm)
    
    return converted_samples, converted_samples.first.sample
