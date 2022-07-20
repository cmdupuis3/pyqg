from abc import ABC, abstractmethod
from functools import cached_property
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)


# Mixin pattern; we could make this abstract and then have multiple FFT types
class KernelFFT:
    """
    Binding layer for Jax FFTs.
    """
    
    def fft(self, x):
        return jnp.fft.rfftn(x, axes=(-2,-1))

    def ifft(self, xfft):
        return jnp.fft.irfftn(xfft, axes=(-2,-1))


class KernelState(KernelFFT):
    """
    A class to track an instantaneous state of the kernel. 
    DO NOT MAKE IT STATEFUL OR MUCH PAIN WILL ENSUE! The 
    statefulness is supposed to quarantined in PSKernel.
    This is structured for lazy evaluation, but calling 
    `state.dqhdt` should calculate everything.
    """
    
    def __init__(self, q, Ubg, a, grid, rek, uv_par, q_par):
        self.q = q
        self.Ubg = Ubg
        self.a = a
        
        self.rek = rek
        self.uv_par = uv_par
        self.q_par = q_par
        
        self.grid = grid
    
    @cached_property
    def ph(self):
        return jnp.apply_over_axes(jnp.sum, self.a * self.qh, [0])

    @cached_property
    def uh(self):
        return jnp.apply_along_axis(jnp.multiply, 1, self.ph, self.grid.il)
        #return jnp.multiply(self.ph, self.grid.il, axis=1)
    
    @cached_property
    def vh(self):
        return jnp.apply_along_axis(jnp.multiply, 2, self.ph, self.grid.ik)
        #return jnp.multiply(self.ph, self.grid.ik, axis=2)
    
    @cached_property
    def qh(self):
        return self.fft(self.q)
    
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
    def uqh(self):
        return self.fft(self.uq)
    
    @cached_property
    def vqh(self):
        return self.fft(self.vq)
    
    @cached_property
    def _du_dv(self):
        return self.uv_par()
    
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
        return self.q_par()
    
    @cached_property
    def dqh(self):
        return self.fft(self.dq)
    
    @cached_property
    def dqhdt(self):
        k = self.grid.nz-1
        
        adv = - ( jnp.apply_along_axis(jnp.multiply, 2, self.uqh, self.grid.ik) +
                  jnp.apply_along_axis(jnp.multiply, 1, self.vqh, self.grid.il) +
                  jnp.multiply(self.ph,  self.grid._ikQy, axes = [0,2]) # check axes!
                )
        uv_par = lambda x: x + ( jnp.apply_along_axis(jnp.multiply, 2, self.dvh, self.grid.ik) -
                                 jnp.apply_along_axis(jnp.multiply, 1, self.duh, self.grid.il) )
        q_par  = lambda x: x + self.dqh
        fric   = lambda x: x + self.rek*(self.grid.k2l2 * self.ph[k])
        
        def compose(f1, f2):
            return lambda x: f1(f2(x))
        
        # This is written this way so jax will hopefully see a more inlined version 
        # of what we're doing, potentially avoiding intermediate sums
        if self.rek:
            if uv_parameterization is not None:
                if q_parameterization is not None:
                    return compose(compose(uv_par, q_par), fric)(adv)
                else:
                    return compose(uv_par, fric)(adv)
            elif q_parameterization is not None:
                return compose(q_par, fric)(adv)
            else:
                return fric(adv)
        else:
            if uv_parameterization is not None:
                if q_parameterization is not None:
                    return compose(uv_par, q_par)(adv)
                else:
                    return uv_par(adv)
            elif q_parameterization is not None:
                return q_par(adv)
            else:
                return adv

class PSKernel(KernelFFT):

    def _empty_com(self):
        return jnp.zeros((self.grid.nz, self.grid.nl, self.grid.nk), jnp.complex64) # complex128
    
    def __init__(self, q, Ubg, a, grid, rek = 0.0, uv_par = None, q_par = None):
        self.Ubg = Ubg
        self.a = a
        self.grid = grid
        self.state = KernelState(q, Ubg, a, grid, rek, uv_par, q_par)
        
        # time stuff
        self.dt = 0.0
        self.t  = 0.0
        self.tc = 0
        self.ablevel = 0
        
        # the tendency
        self.dqhdt    = self._empty_com() # need zeros to start 
        self.dqhdt_p  = self._empty_com()
        self.dqhdt_pp = self._empty_com()
        
    @property
    def q(self):
        return self.state.q
        
    @property
    def qh(self):
        return self.state.qh
    
    def _forward_timestep(self):
        """Step forward based on tendencies"""

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

        # do FFT of new qh
        self.state = KernelState(self.q, self.Ubg, self.a, self.grid)
        self.dqhdt = self.state.dqhdt
        self.q = self.state.q # don't think we need this, but idk

        self.tc += 1
        self.t += self.dt
        
        return
    
    def call(self, diagnostics):
        _ = self.state.dqhdt
        diagnostics()
        self._forward_timestep()
        return
        
        
    
# def tendency_forward_euler(dt, dqdt):
#     """Compute tendency using forward euler timestepping."""
#     return dt * dqdt

# def tendency_ab2(dt, dqdt, dqdt_p):
#     """Compute tendency using Adams Bashforth 2nd order timestepping."""
#     DT1 = 1.5*dt
#     DT2 = -0.5*dt
#     return DT1 * dqdt + DT2 * dqdt_p

# def tendency_ab3(dt, dqdt, dqdt_p, dqdt_pp):
#     """Compute tendency using Adams Bashforth 3nd order timestepping."""
#     DT1 = 23/12.*dt
#     DT2 = -16/12.*dt
#     DT3 = 5/12.*dt
#     return DT1 * dqdt + DT2 * dqdt_p + DT3 * dqdt_pp