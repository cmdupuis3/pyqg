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


# Mixin pattern; we could make this abstract and then have multiple FFT types
class KernelFFT:
    
    def fft(self, x):
        return jnp.fft.rfftn(x, axes=(-2,-1))

    def ifft(self, x):
        return jnp.fft.irfftn(x, axes=(-2,-1))
