from dataclasses import dataclass, field
from functools import cached_property
import numpy as np


@dataclass
class Grid:
    """
    Set up spatial and spectral grids and related constants.
    
    
        .. note:: All of the test cases use ``nx==ny``. Expect bugs if you choose
                  these parameters to be different.
        nx : int
            Number of grid points in the x direction.
        ny : int
            Number of grid points in the y direction (default: nx).
        L : number
            Domain length in x direction. Units: meters.
        W :
            Domain width in y direction. Units: meters (default: L).
    
    """
    nz: int
    ny: int # grid resolution
    nx: int
    
    L: float # domain size is L [m]
    W: float
    
    
    x: int = field(init=False)
    y: int = field(init=False)
    
    dk: float = field(init=False)
    dl: float = field(init=False)
    
    k:  int = field(init=False)
    l:  int = field(init=False)
    kk: int = field(init=False)
    ik: int = field(init=False)
    ll: int = field(init=False)
    il: int = field(init=False)
    
    M:  int = field(init=False)
    
    wv2:  float = field(init=False)
    wv:   float = field(init=False)
    wv2i: float = field(init=False)
    
    nk:   int = field(init=False)
    nl:   int = field(init=False)
    k2l2: int = field(init=False)
    
    # TODO: double check this; may vary by _initialize_background implementation
    def set__ikQy(self, Qy): 
        self._ikQy = (1j * (np.asarray(self.kk)[np.newaxis, :] *
                            np.asarray(Qy)[:, np.newaxis]))
    
    def __post_init__(self):
        
        if self.ny is None:
            self.ny = self.nx
        if self.W is None:
            self.W = self.L

        self.x,self.y = np.meshgrid(
            np.arange(0.5,self.nx,1.)/self.nx*self.L,
            np.arange(0.5,self.ny,1.)/self.ny*self.W )

        # Notice: at xi=1 U=beta*rd^2 = c for xi>1 => U>c
        # wavenumber one (equals to dkx/dky)
        self.dk = 2.*np.pi/self.L
        self.dl = 2.*np.pi/self.W

        # wavenumber grids
        self.nl = self.ny
        self.nk = int(self.nx/2+1)
        self.ll = self.dl*np.append( np.arange(0.,self.nx/2),
            np.arange(-self.nx/2,0.) )
        self.kk = self.dk*np.arange(0.,self.nk)

        self.k, self.l = np.meshgrid(self.kk, self.ll)
        self.ik = 1j*self.k
        self.il = 1j*self.l
        # physical grid spacing
        self.dx = self.L / self.nx
        self.dy = self.W / self.ny

        # constant for spectral normalizations
        self.M = self.nx*self.ny

        # isotropic wavenumber^2 grid
        # the inversion is not defined at kappa = 0
        self.wv2 = self.k**2 + self.l**2
        self.wv = np.sqrt( self.wv2 )

        iwv2 = self.wv2 != 0.
        self.wv2i = np.zeros_like(self.wv2)
        self.wv2i[iwv2] = self.wv2[iwv2]**-1
        
        
        # Kernel grid calculations
        self.k2l2 = np.zeros((self.nl, self.nk))
        for j in range(self.nl):
            for i in range(self.nk):
                self.k2l2[j,i] = self.kk[i]**2 + self.ll[j]**2
                
    