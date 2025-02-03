import dimod 
from dwave.cloud import Client
from typing import Optional, Dict, Tuple
from dwave.system.samplers import DWaveSampler
# from dwave.samplers import TabuSampler, SteepestDescentSolver
from dwave.system.composites import EmbeddingComposite
import os
import json
import time
import zipfile
from dwave.embedding.chain_strength import uniform_torque_compensation, scaled
from dwave.samplers import (TabuSampler, SteepestDescentSolver,
                             RandomSampler, ExactSolver, SimulatedAnnealingSampler)

class dimod_optimizer_local:
        def __init__(self,
                     optimizer_type:Optional[str]='SimulatedAnnealing')-> None:
            
            if optimizer_type == 'SimulatedAnnealing':
                self.sampler = SimulatedAnnealingSampler()
            elif optimizer_type == 'Tabu':
                self.sampler = TabuSampler()
            elif optimizer_type == 'SteepestDescent':
                self.sampler = SteepestDescentSolver()
            elif optimizer_type == 'RandomSampler':
                self.sampler = RandomSampler()
            elif optimizer_type == 'ExactSolver':
                self.sampler = ExactSolver()
            else:
                 raise ValueError(f'unknown optimizer: {optimizer_type}')
        
        def sample_qubo(self, dwave_qubo: Dict[Tuple[int, int], float], num_reads:int, chain_strength: float) -> Tuple[dimod.sampleset.SampleSet,
                                                                                                                        Dict[int, int]]:
            # bqm = dimod.BinaryQuadraticModel.from_qubo(dwave_qubo)
            sampleset =  self.sampler.sample_qubo(dwave_qubo, 
                                                  num_reads=num_reads,
                                                  chain_strength=chain_strength)
            
            return sampleset, sampleset.first.sample
        
        def sample_ising(self, linear_terms: Dict[int, float],
                          quadratic_terms: Dict[Tuple[int, int], float], num_reads:int, chain_strength: float) -> Tuple[dimod.sampleset.SampleSet,
                                                                                                                        Dict[int, int]]:
            sampleset = self.sampler.sample_ising(linear_terms, quadratic_terms,
                                                  num_reads=num_reads,
                                                  chain_strength=chain_strength)
            return sampleset, sampleset.first.sample


class dimod_optimizer_cloud(dimod_optimizer_local):
        def __init__(self,
                     DWAVE_API_TOKEN:str,
                     reverse_anneal_params: Optional[dict] = dict())-> None:
        
            self.sampler = None
            self.DWAVE_API_TOKEN= DWAVE_API_TOKEN
            self.reverse_anneal_params = reverse_anneal_params


        def sample_qubo(self, dwave_qubo: Dict[Tuple[int, int], float], 
                        num_reads:int, min_qubits_needed:int,  
                        chain_strength: float) -> Tuple[dimod.sampleset.SampleSet,
                                                                                                                                               Dict[int, int]]:


            with Client.from_config(token=self.DWAVE_API_TOKEN) as client:
                    dwave_solvers = [ (solver, solver.num_active_qubits) for solver in 
                                     client.get_solvers(refresh=True, online=True) if ('qubo' in solver.supported_problem_types) ]
                    ### get solver with most qubits!
                    dwave_solver, n_qubits = max(dwave_solvers, key=lambda x: x[1])
                    assert min_qubits_needed <= n_qubits, 'too many qubits for required problem'
                    
                    self.sampler = EmbeddingComposite(DWaveSampler(dwave_solver))
                    self.solver_properties = dwave_solver.properties

                    ### do sampling
                    sampleset = self.sampler.sample_qubo(dwave_qubo,
                                                          num_reads=num_reads, 
                                                          return_embedding=True,
                                                          chain_strength=chain_strength,
                                                           **self.reverse_anneal_params)
                ## note this will close the connection once samples are done!!! (context manager)
                    
            return sampleset, sampleset.first.sample


def save_annealing_results(anneal_sampleset: dimod.sampleset.SampleSet,
                           file_save_loc:Optional[str]=None, 
                           extra_ending:Optional[str]='',
                           verbose:Optional[bool]=True,
                           extra_dictionary:Optional[Dict]=dict(),
                           zip_data:Optional[Dict]=True):
    
    if file_save_loc is None:
        file_save_loc = os.getcwd()

    assert os.path.isdir(file_save_loc), 'save location is not a valid dir'

    base_name_zip  = f'{extra_ending}_' + time.strftime("%Y%m%d-%H%M%S") + '.zip'
    base_name_json = f'{extra_ending}_' + time.strftime("%Y%m%d-%H%M%S") + '.json'
    if zip_data:
        file_path = os.path.join(file_save_loc ,
                                    base_name_zip)

        with zipfile.ZipFile(file_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file: 
            output_data = anneal_sampleset.to_serializable()
            output_data.update(extra_dictionary)
            # Dump JSON data
            dumped_JSON: str = json.dumps(output_data, indent=6)  ## ensure_ascii=False,
            # Write the JSON data into `data.json` *inside* the ZIP file
            zip_file.writestr(base_name_json, data=dumped_JSON)
            # Test integrity of compressed archive
            zip_file.testzip()
    else:
        file_path = os.path.join(file_save_loc ,
                                   base_name_json)

        with open(file_path, mode="w") as outfile: 
            output_data = anneal_sampleset.to_serializable()
            output_data.update(extra_dictionary)
            json.dump(output_data, outfile, indent=6)

    if verbose:
        print(f'saved data at: {file_path}')

    return file_path


def get_chain_strength(source_bqm: dimod.BinaryQuadraticModel, 
                       prefactor:Optional[float]=1.414, method:Optional[str]='uniform_torque_compensation',
                       embedded_bqm:Optional[dimod.BinaryQuadraticModel]=None) -> float:
    """
     https://arxiv.org/pdf/2007.01730
    https://dwave-systemdocs.readthedocs.io/en/master/reference/embedding.htm
    """
    if method == 'uniform_torque_compensation':
          chain_strength = scaled(source_bqm, embedding=embedded_bqm,
                                   prefactor=prefactor)
    elif method == 'uniform_torque_compensation':
          chain_strength = uniform_torque_compensation(source_bqm, embedding=embedded_bqm, 
                                                       prefactor=prefactor)
    else:
        raise ValueError(f'unknown method: {method}')

    return chain_strength


def Q_dict_to_bqm(dwave_Q: Dict[Tuple[int,int], float]) -> dimod.BinaryQuadraticModel:

    return dimod.BinaryQuadraticModel.from_qubo(dwave_Q)

