from abc import ABC, abstractmethod
from functools import cached_property
from dataclasses import dataclass

@dataclass
class KernelGrid:
    nz: int
    ny: int
    nx: int
    nl: int
    
    kk: int
    ik: int
    ll: int
    il: int
    k2l2: int
    
    ikQy: int
    
    nk: int = field(init=False)
    
    def __post_init__():
        self.nk = int(self.nx/2 +1)