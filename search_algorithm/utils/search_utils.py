from generate import respond_to_batch
from dataclasses import dataclass
import torch.nn as nn
import torch

@dataclass
class SearchUtils:
    policy: nn.Module
    rollout_policy: nn.Module
    
    def Expansion(
        self, parent_state: torch.LongTensor, model_max_length: int, top_k: int, top_p: int, temperature: int,
        termination_token: torch.LongTensor
    ):
        child_state = respond_to_batch(
            model=self.policy, 
            termination_token=termination_token,
            queries=parent_state,
            model_max_length=model_max_length,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )
        
        return child_state
    
    def Rollout(
        self, child_state: torch.LongTensor, model_max_length: int, top_k: int, top_p: int, temperature: int, 
        termination_token: torch.LongTensor
    ):
        rollout_state = respond_to_batch(
            model=self.policy, 
            termination_token=termination_token,
            queries=child_state,
            model_max_length=model_max_length,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )
        
        return rollout_state
    
    def QTable(self):
        return None
    
    def Selection(self):
        return None
    
    def VerifierScore(self):
        return None
        
