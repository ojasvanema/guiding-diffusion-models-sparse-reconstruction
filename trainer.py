import yaml
import os
from pathlib import Path
import torch
import argparse, time, logging
from IPython.display import clear_output
from PIL import Image
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from conflictfree.grad_operator import ConFIGOperator
from conflictfree.utils import get_gradient_vector
from conflictfree.length_model import ProjectionLength
from my_config_length import UniProjectionLength

from utils import *
from diffusion import Diffusion
from networks import Model
from datasets import KolmogorovFlowDataset
from residuals import ResidualOp


LOGGER = None


def load_config(args):
    if bool(args.dataset):
        config_path = "../../configs/kmflow_re1000_rs256.yml"
    else:
        config_path = "../../configs/kmflow_re1000_rs160.yml"

    with open(Path(config_path), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    if args.device > -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device('cpu')

    seed = 1234

    config.device = device
    config.seed = seed
    return config


def define_folder(folder=None):
    base_path = Path("runs")
    if folder:
        os.chdir(base_path / str(folder))
        return
    
    for i in range(1000):
        new_fold = base_path / str(i).zfill(3)
        if not new_fold.exists():
            os.mkdir(new_fold)
            os.chdir(new_fold)
            return new_fold


def parse_args():
    parser = argparse.ArgumentParser(description='CmdLine Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', default=10, type=int, help='training epochs')
    parser.add_argument('--batch', default=1, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--gamma', default=0.95, type=float, help='Gamma lr scheduler')
    parser.add_argument('--device', default=0, type=int, help='Index of GPU to use for training')
    parser.add_argument('--loss_m', default="l1", type=str, help='Residual loss method: l1 or l2')
    parser.add_argument('--dataset', default=1, type=int, help='What dataset to use for training. 1: shu, 0: group dataset')
    parser.add_argument('--eq_res', default=-1, type=float, help='Coefficient of the Vorticity equation residual contribution. If -1, means dynamic')
    parser.add_argument('--method', default="PINN", type=str, help='Training method. Options: std, PINN, ConFIG, multiConFIG')
    parser.add_argument('--ndata', default=1000, type=int, help='Dataset size')
    parser.add_argument('--nmulti', default=1, type=int, help='Number of steps over which to perform multiConFIG method')
    parser.add_argument('--last_lr', default=1e-5, type=float, help='Last lr at the end of the scheduler.')
    parser.add_argument('--validation', default=True, type=bool, help='Compute validation')
    parser.add_argument('--checkpoint', default="", type=str, help='Path of checkpoint from where to continue training.')
    parser.add_argument('--length', default=0, type=int, help='Method to compute length of ConFIG gradient. 0: proj, 1: uniProj')
    parser.add_argument('--seed', default=-1, type=int, help='Seed value')
    
    args = parser.parse_args()
    
    params = {}
    params.update(vars(args))
    log(f"Params: {params}")

    return args


def set_logger():
    global LOGGER

    LOGGER = logging.getLogger()
    LOGGER.addHandler(logging.StreamHandler())
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.FileHandler('./run.log'))


def log(s):
    global LOGGER
    # print(s)
    LOGGER.info(s)
  

def compute_validation(model, diffusion, dataset, device, residual_op, noise=None):
    model.eval()
    nsamples = noise.shape[0]
    l1_loss = np.zeros(nsamples) 
    with torch.no_grad():
        for i in range(nsamples):
            print("Sample", i)
            x_noise = noise[i].unsqueeze(0).to(device)
            y_pred = diffusion.ddpm(x_noise, model, 1000, plot_prog=False)
            res, _ = residual_op(dataset.scaler.inverse(y_pred))    
            l1_loss[i] = torch.mean(abs(res))

    model.train()
    return np.mean(l1_loss)


def diffusion_standard_step(model, diffusion, y, scaler, optimizer, loss_m, residual_op, **kargs):
    batch_size = y.shape[0]
    t = torch.randint(0, diffusion.num_timesteps, size=(batch_size,), device=y.device)

    x_t, noise = diffusion.forward(y, t)
    e_pred = model(x_t, t)
    mse_loss = (noise - e_pred).square().mean()

    mse_loss.backward()
    optimizer.step()

    # Compute res_loss for metrics comparison
    with torch.no_grad():
        a_b = diffusion.alphas_b[t].view(batch_size, 1, 1, 1)
        x0_pred = (x_t - (1 - a_b).sqrt() * e_pred) / a_b.sqrt()
        eq_residual, _ = residual_op(scaler.inverse(x0_pred))
        eq_res_m = loss_m(eq_residual)

    return mse_loss, eq_res_m


def diffusion_PINN_step(model, diffusion, y, scaler, optimizer, loss_m, residual_op, **kargs):
    batch_size = y.shape[0]
    t = torch.randint(0, diffusion.num_timesteps, size=(batch_size,), device=y.device)

    x_t, noise = diffusion.forward(y, t)
    e_pred = model(x_t, t)
    mse_loss = (noise - e_pred).square().mean()

    a_b = diffusion.alphas_b[t].view(batch_size, 1, 1, 1)
    x0_pred = (x_t - (1 - a_b).sqrt() * e_pred) / a_b.sqrt()
    eq_residual, _ = residual_op(scaler.inverse(x0_pred))

    eq_res_m = loss_m(eq_residual)

    with torch.no_grad():
        if kargs["eq_res"] < 0:
            coef = mse_loss / eq_res_m
        else:
            coef = kargs["eq_res"]

    loss = mse_loss + coef * eq_res_m

    loss.backward()
    optimizer.step()

    return mse_loss, eq_res_m


def diffusion_ConFIG_step(model, diffusion, y, scaler, optimizer, loss_m, residual_op, **kargs):
    batch_size = y.shape[0]
    t = torch.randint(0, diffusion.num_timesteps, size=(batch_size,), device=y.device)

    x_t, noise = diffusion.forward(y, t)
    e_pred = model(x_t, t)
    mse_loss = (noise - e_pred).square().mean()

    a_b = diffusion.alphas_b[t].view(batch_size, 1, 1, 1)
    x0_pred = (x_t - (1 - a_b).sqrt() * e_pred) / a_b.sqrt()
    eq_residual, _ = residual_op(scaler.inverse(x0_pred))
    residual_loss = loss_m(eq_residual)
    residual_loss_unscaled = residual_loss.clone()
    mse_loss.backward(retain_graph=True)
    grads_1 = get_gradient_vector(model, y.device)
    optimizer.zero_grad()

    # with torch.no_grad():
    #     coef = mse_loss / residual_loss
    # residual_loss = residual_loss * coef
    
    residual_loss.backward()
    grads_2 = get_gradient_vector(model, y.device)

    # with torch.no_grad():
    #     if kargs["eq_res"] < 0: # Dynamic coeff 
    #         grads_2 = grads_2 * (grads_1.norm() / grads_2.norm()) 
    #     else:
    #         grads_2 = grads_2 * kargs["eq_res"]

    # kargs["config_operator"].update_gradient(model, torch.stack([grads_1, grads_2], 0))
    kargs["config_operator"].update_gradient(model, [grads_1, grads_2])
    optimizer.step()
    
    return mse_loss, residual_loss_unscaled


def diffusion_multi_ConFIG_step(model, diffusion, y, scaler, optimizer, loss_m, residual_op, **kargs):
    batch_size = y.shape[0]
    n_groups = kargs["nmulti"]
    gradiends = []

    group_size = diffusion.num_timesteps // n_groups

    mse_losses = np.zeros(n_groups)
    res_losses = np.zeros(n_groups)

    for i in range(n_groups):
        optimizer.zero_grad()
        
        begin = group_size * i
        end   = group_size * (i + 1) if i < n_groups - 1 else diffusion.num_timesteps
        t = torch.randint(begin, end, size=(1,), device=y.device).repeat(batch_size)

        x_t, noise = diffusion.forward(y, t)
        e_pred = model(x_t, t)
        mse_loss = (noise - e_pred).square().mean()
        mse_losses[i] = mse_loss.item()
        retain_graph =  i < n_groups - 1
        mse_loss.backward(retain_graph=retain_graph)
        
        grads = get_gradient_vector(model, y.device)
        gradiends.append(grads)

        # Evaluating residual loss
        a_b = diffusion.alphas_b[t].view(batch_size, 1, 1, 1)
        x0_pred = (x_t - (1 - a_b).sqrt() * e_pred) / a_b.sqrt()
        eq_residual, _ = residual_op(scaler.inverse(x0_pred))
        residual_loss = loss_m(eq_residual)        
        res_losses[i] = residual_loss

    kargs["config_operator"].update_gradient(model, torch.stack(gradiends, 0))
    optimizer.step()
    
    return np.mean(mse_losses), np.mean(res_losses)


def train(config, train_dataset, scaler, args=None):
    model = Model(config)
    model.to(config.device)
    model.train()

    diffusion = Diffusion(config)
    writer = SummaryWriter(".")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    start_epoch = 0

    validation_noise = torch.randn((10, 3, 256, 256))

    if args.checkpoint:
        cp = torch.load(args.checkpoint, weights_only=True)
        model.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        scheduler.load_state_dict(cp['scheduler_state_dict'])
        start_epoch = cp['epoch'] + 1
        print(f"Starting from checkpoint {args.checkpoint} at epoch {start_epoch}")

    config_operator = ConFIGOperator(length_model={
        0: ProjectionLength(),
        1: UniProjectionLength()
    }[args.length])

    residual_op = ResidualOp(device=config.device)

    loss_m = {
        "l1": lambda x: torch.mean(torch.abs(x)),
        "l2": lambda x: torch.sqrt(torch.mean(x**2))
    }[args.loss_m]

    train_step = {
        "PINN": diffusion_PINN_step,
        "ConFIG": diffusion_ConFIG_step,
        "std": diffusion_standard_step,
        "multiConFIG": diffusion_multi_ConFIG_step
    }[args.method]

    for epoch in range(start_epoch, args.epochs):
        residual_loss = 0
        mse_loss      = 0

        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            model.train()
            x = data.to(config.device)

            e_loss, res_loss = train_step(
                model, diffusion, x, 
                scaler, optimizer, loss_m, residual_op,
                config_operator=config_operator, eq_res=args.eq_res, nmulti=args.nmulti
            )

            residual_loss += res_loss
            mse_loss      += e_loss
            
            log(f"[{epoch}, {i}]: eq_res {res_loss:.4f}, mse {e_loss:.4f}")

        if optimizer.param_groups[0]['lr'] > args.last_lr:
            scheduler.step()
            log(f"(Scheduler) new lr: {optimizer.param_groups[0]['lr']}")
        else:
            optimizer.param_groups[0]['lr'] = args.last_lr
            log(f"(Scheduler) last lr: {optimizer.param_groups[0]['lr']}")

        if (epoch + 1) % 25 == 0:
            log(f"Saving model {epoch + 1}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'residual_loss': residual_loss,
                'mse_loss': mse_loss,
            }, f"checkpoint{epoch+1}.pt")

            if args.validation:
                log("Computing validation...")
                val_loss = compute_validation(model, diffusion, train_dataset, config.device, residual_op, noise=validation_noise)
                writer.add_scalar('Val - l1 residual loss', val_loss, epoch)
                log(f"Validation loss: {val_loss}")

        writer.add_scalar('MSE error', e_loss / len(train_dataloader), epoch)
        writer.add_scalar('Vorticity eq residual', residual_loss / len(train_dataloader), epoch)
        
        if epoch % 5 == 0: writer.flush()


def main():
    time_start = time.time()

    new_fold = define_folder()
    set_logger()
    log(f"Run folder: {new_fold}")

    args = parse_args()
    config = load_config(args)

    log(f"Process ID: {os.getpid()}")
    log(f"Filename: {__file__}")

    log("Loading dataset...")
    dataset = KolmogorovFlowDataset(data_folder="../../data", shu=bool(args.dataset), seed=1234, size=args.ndata)
    # train_dataset, _ = dataset.split()
    train_dataset = dataset

    log("Dataset loaded")
    log(f"Dataset length: {len(train_dataset)}")

    if args.seed == -1: 
        args.seed = int(time.time() * 100) % 2**32

    fix_randomness(args.seed)
    log(f"Training seed: {args.seed}")

    train(config, train_dataset, dataset.scaler, args=args)

    time_end = time.time()
    log(f"Training took {time_end - time_start}")


if __name__ == "__main__":
    main()