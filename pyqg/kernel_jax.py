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


class KernelState(KernelFFT):
    
    def __init__(self, qh, Ubg, a, grid):
        self.qh = qh
        self.Ubg = Ubg
        self.a = a
        self.grid = grid
    
    @cached_property
    def ph(self):
        return jnp.apply_over_axes(jnp.sum, self.a * self.qh, [0])

    @cached_property
    def uh(self):
        return jnp.multiply(self.ph, self.grid.il, axis=1)
    
    @cached_property
    def vh(self):
        return jnp.multiply(self.ph, self.grid.ik, axis=2)
    
    @cached_property
    def q(self):
        return self.ifft(self.qh)
    
    @cached_property
    def u(self):
        return self.ifft(self.uh)
    
    @cached_property
    def v(self):
        return self.ifft(self.vh)
    
    @cached_property
    def uq(self):
        return (self.u + self.Ubg) * self.q
    
    @cached_property
    def vq(self):
        return self.v * self.q
    
    @cached_property
    def _du_dv(self):
        return self.uv_parameterization(self)
    
    @cached_property
    def du(self):
        return _du_dv[0]
    
    @cached_property
    def dv(self):
        return _du_dv[1]
    
    @cached_property
    def duh(self):
        return self.fft(self.du)
    
    @cached_property
    def dvh(self):
        return self.fft(self.dv)
    
    @cached_property
    def dq(self):
        return self.q_parameterization(self)
    
    @cached_property
    def dqh(self):
        return self.fft(self.dq)
    
    @cached_property
    def dqhdt(self):
        k = self.nz-1
        
        adv = - ( jnp.multiply(self.uqh, self.grid.ik, axis=2) +
                  jnp.multiply(self.vqh, self.grid.il, axis=1) +
                  jnp.multiply(self.ph,  self.grid.ikQy, axes = [0,2]) # check axes!
                )
        uv_par = lambda x: x + ( jnp.multiply(self.dvh, self.grid.ik, axis=2) -
                                 jnp.multiply(self.duh, self.grid.il, axis=1) )
        q_par  = lambda x: x + self.dqh
        fric   = lambda x: x + self.rek*(self.grid.k2l2 * self.ph[k])
        
        def compose(f1, f2):
            return lambda x: f1(f2(x))
        
        if self.rek:
            if uv_parameterization is not None:
                if q_parameterization is not None:
                    return compose(compose(uv_par, q_par), fric)(adv)
                else
                    return compose(uv_par, fric)(adv)
            elif q_parameterization is not None:
                return compose(q_par, fric)(adv)
            else:
                return fric(adv)
        else:
            if uv_parameterization is not None:
                if q_parameterization is not None:
                    return compose(uv_par, q_par)(adv)
                else
                    return uv_par(adv)
            elif q_parameterization is not None:
                return q_par(adv)
            else:
                return adv

class PSKernel(KernelFFT):
    
    def _empty_real(self):
        """Allocate a space-grid-sized variable for use with fftw transformations."""
        return jnp.zeros((self.nz, self.ny, self.nx), jnp.float32) # float64

    def _empty_com(self):
        """Allocate a Fourier-grid-sized variable for use with fftw transformations."""
        return jnp.zeros((self.nz, self.nl, self.nk), jnp.complex64) # complex128
    
    def __init__(self, qh, Ubg, a, grid):
        self.Ubg = Ubg
        self.a = a
        self.grid = grid
        self.state = KernelState(qh, Ubg, a, grid)
        
        # time stuff
        self.dt = 0.0
        self.t  = 0.0
        self.tc = 0
        self.ablevel = 0
        
        # the tendency
        self.dqhdt    = self._empty_com()
        self.dqhdt_p  = self._empty_com()
        self.dqhdt_pp = self._empty_com()
        
    @property
    def qh(self):
        return self.state.qh
    
    @property
    def dqhdt(self):
        return self.state.dqhdt
    
    def _forward_timestep(self):
        """Step forward based on tendencies"""
        #self.dqhdt = self.dqhdt_adv + self.dqhdt_forc

        # Note that Adams-Bashforth is not self-starting
        if self.ablevel==0:
            # forward euler
            dt1 = self.dt
            dt2 = 0.0
            dt3 = 0.0
            self.ablevel=1
        elif self.ablevel==1:
            # AB2 at step 2
            dt1 = 1.5*self.dt
            dt2 = -0.5*self.dt
            dt3 = 0.0
            self.ablevel=2
        else:
            # AB3 from step 3 on
            dt1 = 23./12.*self.dt
            dt2 = -16./12.*self.dt
            dt3 = 5./12.*self.dt

        self.qh = self.filtr * (
            self.qh +
            dt1 * self.dqhdt +
            dt2 * self.dqhdt_p +
            dt3 * self.dqhdt_pp
        )

        self.dqhdt_pp = self.dqhdt_p
        self.dqhdt_p = self.dqhdt
        #self.dqhdt = 0.0

        # do FFT of new qh
        self.q = self.ifft(self.qh) # this destroys qh, need to assign again

        self.tc += 1
        self.t += self.dt
        
        self.state = KernelState(self.qh, self.Ubg, self.a, self.grid)
        
        return