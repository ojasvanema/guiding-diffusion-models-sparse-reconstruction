import numpy as np
import torch


class ResidualOp:
    def __init__(self, device="cpu", N=256, Re=1000, dt=1/32, L=2*np.pi):
        self.Re = Re
        self.dt = dt
        self.L  = L
        self.N  = N
        self.dx = L / N

        #x = torch.linspace(0, L, N, device=device)
        x = torch.linspace(0, L, N+1, device=device)[:-1]
        _, Y = torch.meshgrid(x, x, indexing="ij")
        self.const_force = (-4 * torch.cos(4 * Y)).reshape(1, 1, N, N)

        # Create the wavevector grid in Fourier space
        self.kx = torch.fft.fftfreq(N, self.dx).reshape(1, 1, N, 1).to(device) * 2 * torch.pi * 1j
        self.ky = torch.fft.fftfreq(N, self.dx).reshape(1, 1, 1, N).to(device) * 2 * torch.pi * 1j    
        
        # Negative Laplacian in Fourier space
        self.lap = (self.kx ** 2 + self.ky ** 2).to(device)
        self.lap[..., 0, 0] = 1.0

    
    def __call__(self, w):
        w_h = torch.fft.fft2(w[:, 1:2], dim=[2, 3])
        psi_h = -w_h / self.lap

        u_h = psi_h * self.ky
        v_h = -psi_h * self.kx
        wx_h = self.kx * w_h
        wy_h = self.ky * w_h
        wlap_h = self.lap * w_h

        u = torch.fft.ifft2(u_h, dim=[2, 3]).real
        v = torch.fft.ifft2(v_h, dim=[2, 3]).real
        wx = torch.fft.ifft2(wx_h, dim=[2, 3]).real
        wy = torch.fft.ifft2(wy_h, dim=[2, 3]).real
    
        wlap = torch.fft.ifft2(wlap_h, dim=[2, 3]).real
        dwdt = (w[:, 2:3] - w[:, 0:1]) / (2 * self.dt)
        advection = u*wx + v*wy
        force = self.const_force - 0.1 * w[:, 1:2]
    
        res = dwdt + advection - (wlap / self.Re) - force

        return res, [dwdt, advection, wlap]
