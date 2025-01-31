import dimod
from minorminer import find_embedding
from dwave import embedding
from typing import List, Dict, Tuple, Optional
import networkx as nx
import numpy as np


def get_problem_graph(dwave_qubo:Dict[Tuple[int, int], float]) -> Tuple[nx.Graph,
                                                                     Dict[int, List[int]]]:
    """
    Args:

    Note to get edge weights use:

    nx.get_edge_attributes(problem_graph, "bias")
    """

    source_bqm_qubo = dimod.BinaryQuadraticModel.from_qubo(dwave_qubo)
    problem_graph = dimod.to_networkx_graph(source_bqm_qubo)

    node_edge_dict = {node: list(problem_graph.neighbors(node)) 
                      for node in problem_graph.nodes}

    return problem_graph, node_edge_dict


def embed_problem_onto_hardware(problem_graph: dimod.BinaryQuadraticModel,
                    hardware_graph: nx.Graph,
                    maxtime_sec:Optional[int]=120,
                    attempts:Optional[int]=10) -> Tuple[Dict[Tuple[int, int], float],
                                              int]:
    
    embedded_problem, valid_flag = find_embedding(problem_graph,
                                                list(hardware_graph.edges), 
                                                return_overlap=True, 
                                                timeout=maxtime_sec,
                                                tries= attempts)
    return embedded_problem, valid_flag


def define_embedded_qubo_problem(embedded_problem: Dict[Tuple[int, int], float],
                                 edge_dict_hardware:dict,
                                 qubo_Q: dict,
                                 chain_strength:Optional[float]=1.0):
    """

    Args:
        embedded_problem (dict): embedded problem (obtained from embed_problem_onto_hardware function)
        edge_dict (dict): dictionary of edges for each node.
        qubo_Q (dict): Q dict of indices as keys and value as coefficient.
        chain_strength (float): Magnitude of the quadratic bias (in SPIN space) applied between variables to form a chain. Note that the energy penalty of chain breaks is 2 * chain_strength.
    """
    trans_qubo = embedding.embed_qubo(qubo_Q,
                                        embedded_problem, 
                                        edge_dict_hardware,
                                        chain_strength=chain_strength)
    

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


def build_reverse_annealing_schedule(starting_state: Dict[int, int], 
                                     source_bqm:dimod.BinaryQuadraticModel,
                                     reverse_schedule: List[List[int]],
                                     reinitialize_state:bool):
    """
    
    Args:
        starting_state (dict): dictionary of rev
        source_bqm (dimod.BinaryQuadraticModel): original BQM problem
    
    """

    assert set(starting_state.keys()) == set(source_bqm.variables), 'starting state does NOT match BQ problem'
    assert set(starting_state.values()) == {0,1}, 'starting_state is not a valid qubo state'

    ### get energy of this state!
    E0 = source_bqm.energy(starting_state)


    ## Put everything in the correct format
    initial_state_sampleset = dimod.SampleSet.from_samples(starting_state, 
                                                        energy=[E0], 
                                                        vartype=dimod.vartypes.Vartype.BINARY ## <-- can change to SPIN if ising model used
                                                        )
    
    ## define annealing schedule
    reverse_anneal_params = dict(anneal_schedule=reverse_schedule, 
                                initial_state=initial_state_sampleset, 
                                reinitialize_state=reinitialize_state)
    
    return reverse_anneal_params


def build_reverse_annealing_schedule_embedding(starting_state_non_embedded:Dict[int, int],
                                                source_bqm:dimod.BinaryQuadraticModel,
                                                reverse_schedule: List[List[int]],
                                                reinitialize_state:bool,
                                                embedded_bqm:dimod.BinaryQuadraticModel,
                                                hardware_graph,
                                                embedded_problem
                                                ):
    """
    Build reverse anneal schedule where input state is to be embedded into a larger problem!

    Note function performs a check to confirm that initial state (not embedded) is correctly mapped
    to the starting state for the embedded problem (which is generated here!)

    Args:
        starting_state (dict): dictionary of rev
        source_bqm (dimod.BinaryQuadraticModel): original BQM problem
        embedded_problem (): output from embed_problem_onto_hardware 
    """

    assert set(starting_state_non_embedded.keys()) == set(source_bqm.variables), 'starting state does NOT match BQ problem'
    assert set(starting_state_non_embedded.values()) == {0,1}, 'starting_state is not a valid qubo state'

    EmbeddedStructure_problem = embedding.EmbeddedStructure(hardware_graph.edges(), 
                                                            embedded_problem)

    ### get energy of this state!
    E0 = source_bqm.energy(starting_state_non_embedded)


    ## Put everything in the correct format
    initial_state_sampleset = dimod.SampleSet.from_samples(starting_state_non_embedded, 
                                                        energy=[E0], 
                                                        vartype=dimod.vartypes.Vartype.BINARY ## <-- can change to SPIN if ising model used
                                                        )
    
    initial_state_embedded = {val: starting_state_non_embedded[key] for key, terms in EmbeddedStructure_problem.items() for val in terms}
    E0_embedded = embedded_bqm.energy(initial_state_embedded)

    embedded_inital_state_sampleset = dimod.SampleSet.from_samples(initial_state_embedded, 
                                                        energy=[E0_embedded], 
                                                        vartype=dimod.vartypes.Vartype.BINARY)

    ## check embedding is correct!
    checker, _ = unembed_samples(embedded_inital_state_sampleset,
                                 embedded_problem, 
                                 source_bqm)
    assert checker.first.sample == initial_state_sampleset.first.sample, 'mapping not correct'

    ## define annealing schedule
    reverse_anneal_params = dict(anneal_schedule=reverse_schedule, 
                                initial_state=embedded_inital_state_sampleset, 
                                reinitialize_state=reinitialize_state)
    
    return reverse_anneal_params


def create_king_graph(rows: int, cols: int):
    """
    Args:
        rows (int):
        cols (int):
    Returns:
        G (nx.Graph): networkx graph.
        node_edge_dict (dict): dictionary of edges for each node.
        pos (dict): position dictionary to plot graph in a grid.

    """
    G = nx.Graph()

    # Add nodes for each cell in the grid
    for index in range(rows * cols):
        G.add_node(index)

    # Add edges for King moves (adjacent including diagonals)
    pos = {}
    for r in range(rows):
        for c in range(cols):
            current_index = r * cols + c
            pos[current_index] = (c, -r)

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # Skip self-loops

                    nr, nc = r + dr, c + dc

                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor_index = nr * cols + nc
                        G.add_edge(current_index, neighbor_index)

    node_edge_dict = {node: list(G.neighbors(node)) for node in G.nodes}

    ### old and slightly slower way

    # G = nx.grid_2d_graph(rows, cols)  # Creates a grid graph
    # G = nx.convert_node_labels_to_integers(G)  # Relabel nodes to start from 0

    # # Add diagonal edges (king moves)
    # pos = {}
    # for i in range(rows):
    #     for j in range(cols):
    #         current = i * cols + j

    #         ## Define grid positions
    #         pos[current] = (j, -i)

    #         ## add edges
    #         for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
    #             ni, nj = i + di, j + dj
    #             if 0 <= ni < rows and 0 <= nj < cols:
    #                 neighbor = ni * cols + nj
    #                 G.add_edge(current, neighbor)

    # node_edge_dict = {node: list(G.neighbors(node)) for node in G.nodes}

    # # # Define grid positions
    # # pos = {i * cols + j: (j, -i) for i in range(rows) for j in range(cols)}

    # # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    # # plt.title("King Graph")
    # return G, node_edge_dict, pos

    return G, node_edge_dict, pos
