class PseudoSpectralKernel:
    
    def _empty_real(self):
        """Allocate a space-grid-sized variable for use with fftw transformations."""
        return jnp.zeros((self.nz, self.ny, self.nx), jnp.float32) # float64

    def _empty_com(self):
        """Allocate a Fourier-grid-sized variable for use with fftw transformations."""
        return jnp.zeros((self.nz, self.nl, self.nk), jnp.complex64) # complex128
    
    def __init__(self, nz, ny, nx, fftw_num_threads=1):
        
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.nl = ny
        self.nk = int(nx/2 + 1)
        self.a     = jnp.zeros((self.nz, self.nz, self.nl, self.nk), jnp.complex128)
        self.kk    = jnp.zeros((self.nk), jnp.float64)
        self._ik   = jnp.zeros((self.nk), jnp.complex128)
        self.ll    = jnp.zeros((self.nl), jnp.float64)
        self._il   = jnp.zeros((self.nl), jnp.complex128)
        self._k2l2 = jnp.zeros((self.nl, self.nk), jnp.float64)
        
        
        
        self.q   = self._empty_real()
        self.qh  = self._empty_com()
        self.ph  = self._empty_com()
        
        self.u   = self._empty_real()
        self.uh  = self._empty_com()
        self.v   = self._empty_real()
        self.vh  = self._empty_com()
        
        self.uq  = self._empty_real()
        self.uqh = self._empty_com()
        self.vq  = self._empty_real()
        self.vqh = self._empty_com()
        
        
        # variables for subgrid parameterizations
        
        # if has_q_param:
        self.dq  = self._empty_real()
        self.dqh = self._empty_com()
        
        # if has_uv_param:
        self.du  = self._empty_real()
        self.dv  = self._empty_real()
        self.duh = self._empty_com()
        self.dvh = self._empty_com()
        

        # dummy variables for diagnostic ffts
        self.dfftin   = self._empty_real()
        self.dfftout  = self._empty_com()
        self.difftin  = self._empty_com()
        self.difftout = self._empty_real()
        
        
        # time stuff
        self.dt = 0.0
        self.t  = 0.0
        self.tc = 0
        self.ablevel = 0

        # friction
        self.rek = 0.0

        # the tendency
        self.dqhdt    = self._empty_com()
        self.dqhdt_p  = self._empty_com()
        self.dqhdt_pp = self._empty_com()


    def fft(self, x):
        return jnp.fft.rfftn(x, axes=(-2,-1))

    def ifft(self, x):
        return jnp.fft.irfftn(x, axes=(-2,-1))
    

    def _invert(self):

        # invert qh to find ph
        self.ph = jnp.apply_over_axes(jnp.sum, self.a * self.qh, [0])

        # calculate spectral velocities
        self.uh = jnp.multiply(self.ph, self._il, axis=1)
        self.vh = jnp.multiply(self.ph, self._ik, axis=2)
        
        #self.ifft_qh_to_q() # necessary now that timestepping is inside kernel
        self.u = self.ifft(self.uh)
        self.v = self.ifft(self.vh)

        return

    def _do_advection(self):

        # multiply to get advective flux in space
        self.uq = (self.u + self.Ubg) * self.q
        self.vq = self.v * self.q
        
        self.uqh = self.fft(self.uq)
        self.vqh = self.fft(self.vq)

        # spectral divergence
        # overwrite the tendency, since the forcing gets called after
        self.dqhdt = -( jnp.multiply(self.uqh, self._ik, axis=2) +
                        jnp.multiply(self.vqh, self._il, axis=1) +
                        jnp.multiply(self.ph,  self._ikQy, axes = [0,2]) # check axes!
                      )
        return

    def _do_uv_subgrid_parameterization(self):
        
        du, dv = self.uv_parameterization(self)

        self.duh = self.fft(self.du)
        self.dvh = self.fft(self.dv)

        self.dqhdt = self.dqhdt + ( jnp.multiply(self.dvh, self._ik, axis=2) -
                                    jnp.multiply(self.duh, self._il, axis=1) )
        return
        
    def _do_q_subgrid_parameterization(self):
        
        dq = self.q_parameterization(self)
        
        self.dqh = self.fft(self.dq)
        
        self.dqhdt = self.dqhdt + self.dqh
        
        return
    
    def _do_friction(self):
        k = self.nz-1
        if self.rek:
            self.dqhdt = self.dqhdt + self.rek*(self._k2l2 * self.ph[k])
        return
    
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

        for k in range(self.nz):
            self.qh[k] = self.filtr * (
                self.qh[k] +
                dt1 * self.dqhdt[k] +
                dt2 * self.dqhdt_p[k] +
                dt3 * self.dqhdt_pp[k]
            )
            
        self.dqhdt_pp = self.dqhdt_p
        self.dqhdt_p = self.dqhdt
        #self.dqhdt = 0.0

        # do FFT of new qh
        self.q = self.ifft(self.qh) # this destroys qh, need to assign again

        self.tc += 1
        self.t += self.dt
        
        return
