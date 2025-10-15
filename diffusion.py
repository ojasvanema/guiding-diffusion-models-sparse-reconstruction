import numpy as np
import torch
from functools import wraps
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
from utils import *


def conditional_no_grad(param_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            use_grad = kwargs.get(param_name, False)
            if use_grad:
                return func(*args, **kwargs)
            else:
                with torch.no_grad():
                    return func(*args, **kwargs)
        return wrapper
    
    return decorator


def plot_iteration(x, i, freq=100):
    if type(x) is torch.Tensor:
        x = x.cpu()

    if i % freq == 0:
        plt.imshow(x, vmin=-2, vmax=2)
        plt.title(f'Iteration {i}')
        clear_output(wait=True)
        plt.show()


class Diffusion():
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        
        betas = np.linspace(
            config.diffusion.beta_start,
            config.diffusion.beta_end,
            config.diffusion.num_diffusion_timesteps,
            dtype=np.float64
        )

        self.betas = torch.tensor(betas).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_b = self.alphas.cumprod(axis=0).to(self.device).to(torch.float32)
        self.alphas_b_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_b[:-1]])

        mask_schedule_sig = config.diffusion.mask_schedule_sig
        mask_schedule = [x**mask_schedule_sig for x in np.linspace(0, 1, self.num_timesteps)]

        mask_schedule = torch.tensor(mask_schedule, dtype=torch.float32)

        self.mask_schedule = mask_schedule.to(self.device)


    @conditional_no_grad('grad')
    def ddpm(self, x_inp, model, t_start=None, plot_prog=False, grad=False):
        n = x_inp.size(0)
        t_start = self.num_timesteps if t_start is None else t_start

        x     = x_inp[:, 0:3]
        Re_ch = x_inp[:, 3:4]

        for i in reversed(range(t_start)):
            t = (torch.ones(n) * i).to(x.device)
            b = self.betas[i]
            a = self.alphas[i]
            a_b = self.alphas_b[i]
            model_inp = torch.cat((x, Re_ch), dim=1) 
            e = model(model_inp, t)

            x = (1 / a.sqrt()) * (x - (b / (1 - a_b).sqrt()) * e) 

            if i > 0:
                # x += b.sqrt() * torch.randn_like(x)
                x += torch.randn_like(x) * (b * (1 - self.alphas_b[i - 1]) /(1 - a_b)).sqrt() 

            if plot_prog: plot_iteration(x[0,0], i)

        return x
    

    @conditional_no_grad('grad')
    def ddim(self, x, model, t_start, reverse_steps, grad=False, **kargs):
        seq = range(0, t_start, t_start // reverse_steps) 
        next_seq = [-1] + list(seq[:-1])
        n = x.size(0)
        
        for i, j in zip(reversed(seq), reversed(next_seq)):
            t = (torch.ones(n) * i).to(x.device)
            a_b = self.alphas_b[i]
            a_next_b = self.alphas_b[j] if i > 0 else torch.ones(1, device=x.device)

            e = model(x, t)

            x0_pred = (x - e * (1 - a_b).sqrt()) / a_b.sqrt()
            x = a_next_b.sqrt() * x0_pred + (1 - a_next_b).sqrt() * e


        return x


    @conditional_no_grad('grad')
    def masked_diffusion(self, x, model, x_low_res, t_start=1000, reverse_steps=100, grad=False, diff_mask=None, **kargs):
        seq = range(0, t_start, t_start // reverse_steps) 
        next_seq = [-1] + list(seq[:-1])
        n = x.size(0)

        for i, j in zip(reversed(seq), reversed(next_seq)):
            t = (torch.ones(n) * i).to(x.device)
            a_b = self.alphas_b[i]
            a_next_b = self.alphas_b[j] if i > 0 else torch.ones(1, device=x.device)
            e = model(x, t)
            x0_pred = (x - e * (1 - a_b).sqrt()) / a_b.sqrt()

            mask_t = diff_mask * self.mask_schedule[i]
            x_masked = x0_pred * (1 - mask_t) + x_low_res * mask_t

            x = a_next_b.sqrt() * x_masked + (1 - a_next_b).sqrt() * e
                
        return x, diff_mask
    

    def forward(self, x_0: torch.Tensor, t: list):
        noise = torch.randn(x_0.shape, device=self.device)
        tmp1 = torch.einsum('i,ijnm->ijnm', self.alphas_b[t].sqrt(), x_0)
        tmp2 = torch.einsum('i,ijnm->ijnm', (1 - self.alphas_b[t]).sqrt(), noise)
        x_t = tmp1 + tmp2

        # x_t = self.alphas_b[t].sqrt() * x_0 + (1 - self.alphas_b[t]).sqrt() * noise
        return x_t.float(), noise
