import dimod 
from dwave.cloud import Client
from typing import Optional, Dict, Tuple
from dwave.system.samplers import DWaveSampler
# from dwave.samplers import TabuSampler, SteepestDescentSolver
from dwave.system.composites import EmbeddingComposite
import os
import json
import time


class dimod_optimizer_local:
        def __init__(self,
                     optimizer_type:Optional[str]='SimulatedAnnealing')-> None:
            
            if optimizer_type == 'SimulatedAnnealing':
                self.sampler = dimod.SimulatedAnnealingSampler()
            elif optimizer_type == 'Tabu':
                self.sampler = dimod.TabuSampler()
            elif optimizer_type == 'SteepestDescent':
                self.sampler = dimod.SteepestDescentSolver()
            elif optimizer_type == 'RandomSampler':
                self.sampler = dimod.RandomSampler()
            else:
                 raise ValueError(f'unknown optimizer: {optimizer_type}')
        
        def sample_qubo(self, dwave_qubo: Dict[Tuple[int, int], float], num_reads:int) -> None:
            sampleset =  self.sampler.sample_qubo(dwave_qubo, 
                                                  num_reads=num_reads, return_embedding=True)
            return sampleset, sampleset.first.sample
        
        def sample_ising(self, linear_terms: Dict[int, float],
                          quadratic_terms: Dict[Tuple[int, int], float], num_reads:int) -> None:
            sampleset = self.sampler.sample_ising(linear_terms, quadratic_terms,
                                                  num_reads=num_reads, return_embedding=True)
            return sampleset, sampleset.first.sample


class dimod_optimizer_cloud(dimod_optimizer_local):
        def __init__(self,
                     DWAVE_API_TOKEN:str)-> None:
        
            self.sampler = None
            self.DWAVE_API_TOKEN= DWAVE_API_TOKEN
            

        def sample_qubo(self, dwave_qubo: Dict[Tuple[int, int], float], num_reads:int, min_qubits_needed:int) -> None:


            with Client.from_config(token=self.DWAVE_API_TOKEN) as client:
                    dwave_solvers = [ (solver, solver.num_active_qubits) for solver in 
                                     client.get_solvers(refresh=True, online=True) if ('qubo' in solver.supported_problem_types) ]
                    ### get solver with most qubits!
                    dwave_solver, n_qubits = max(dwave_solvers, key=lambda x: x[1])
                    assert min_qubits_needed <= n_qubits, 'too many qubits for required problem'
                    
                    self.sampler = EmbeddingComposite(DWaveSampler(dwave_solver))
                    self.solver_properties = dwave_solver.properties

                    ### do sampling
                    sampleset = self.sampler.sample_qubo(dwave_qubo, num_reads=num_reads, return_embedding=True, **self.reverse_anneal_params)
                ## note this will close the connection once samples are done!!! (context manager)
                    
            return sampleset, sampleset.first.sample


def save_annealing_results(anneal_sampleset: dimod.sampleset.SampleSet,
                           file_save_loc:Optional[str]=None, 
                           extra_ending:Optional[str]='',
                           verbose:Optional[bool]=True,
                           extra_dictionary:Optional[Dict]=dict()):
    
    if file_save_loc is None:
        file_save_loc = os.getcwd()

    base_name = f'{extra_ending}_' + time.strftime("%Y%m%d-%H%M%S") + '.json'
    file_path = os.path.join(file_save_loc ,
                                 base_name)

    with open(file_path, mode="w") as outfile: 
        output_data = anneal_sampleset.to_serializable()
        output_data.update(extra_dictionary)
        json.dump(output_data, outfile, indent=6)
    
    if verbose:
        print(f'saved data at: {file_path}')

    return None
