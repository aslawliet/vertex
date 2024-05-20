from .utils import Expansion, Rollout, VerifierScore, Selection
from .optimisation import GetParameters, QTable, QLearning, QReset, Alpha
from dataclasses import dataclass
import transformers
import torch

@dataclass
class VertexSearch:
    policy_model: transformers
    rollout_policy_model: transformers
    question_tokens: torch.tensor
    bsbl: int = 6 # breadth_search_breadth_limit
    fsbl: int = 6 # forward_search_breadth_limit
    
    def initial_parameters(self):
        self.policy_params = GetParameters(self.policy_model)
        self.rollout_policy_params = GetParameters(self.rollout_policy_model)
    
    def ForwardSearch(self, child_state: torch.tensor):
        forward_qvalues = []
        initial_params = self.rollout_policy_params
        for rollout in range(self.fsbl):
            rollout_sample = Rollout(self.rollout_policy_model, child_state)
            rollout_qvalue = VerifierScore("outcome", self.policy_model, rollout_sample)
            forward_qvalues.append(rollout_qvalue)
            
            QLearning(self.rollout_policy_model, child_state, rollout_sample, rollout_qvalue)
        QReset(self.rollout_policy_model, initial_params)
        
        forward_qvalue = sum(forward_qvalues) / len(forward_qvalues)
        
        return forward_qvalue
    
    def BreadthSearch(self, parent_state: torch.tensor):
        breadth_q_table = QTable()
        initial_params = self.policy_params
        for expansion in range(self.bsbl):
            child_state = Expansion(self.policy_model, self.policy_model.config.termination_token, parent_state)
            forward_qvalue = self.ForwardSearch(child_state=child_state)
            child_qvalue = VerifierScore("process", self.policy_model, child_state)
            alpha_qvalue = Alpha(child_qvalue, forward_qvalue)
            breadth_q_table.update(child_state, alpha_qvalue)
            
            QLearning(self.policy_model, parent_state, child_state, alpha_qvalue)
        QReset(self.policy_model, initial_params)   
        
        return breadth_q_table      
            
    def Search(self):
        seacrh_endtoken = self.policy_model.config.search_endtoken
        root_state = self.question_tokens
        parent_state = root_state
        search = True
        while search:
            breadth_q_table = self.BreadthSearch(parent_state=parent_state)
            optimal_child_state = Selection(breadth_q_table=breadth_q_table)
            
            parent_state = optimal_child_state
            if seacrh_endtoken in parent_state:
                search = False
                
        solution = parent_state
        
        return solution
