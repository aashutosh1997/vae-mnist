import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from models import VAE, VariationalMeanField
import torch.utils.tensorboard as tb

if __name__ == '__main__':
    train_logger = tb.SummaryWriter('logs/train', flush_secs=1)
    valid_logger = tb.SummaryWriter('logs/valid', flush_secs=1)
    dataset = MNIST('data/', download=True, transform=ToTensor())
    train_loader = DataLoader(dataset)    
    dataset = MNIST('data/',train=False, transform=ToTensor)
    test_loader = DataLoader(dataset)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    network = VAE().to(device)
    variational = VariationalMeanField().to(device)
    
    optimizer = torch.optim.Adam(
        list(network.parameters()) + list(variational.parameters()),
        lr = 1e-3,
        betas = (0.9, 0.99)
    )
    
    
    for itr in range(100):
        network.train()
        total_elbo = 0.0
        for idx, (img, lbl) in enumerate(train_loader):
            img = img.to(device)
            lbl = lbl.to(device)
            network.zero_grad()
            variational.zero_grad()
            z, log_q_z = variational(img)
            log_p_xz = network(z, img)
            elbo = log_p_xz-log_q_z
            loss = -elbo.sum(0)
            loss.backward()
            optimizer.step()
            total_elbo += elbo.detach().cpu().numpy().mean()
        train_logger.add_scalar('elbo', total_elbo/len(train_loader.dataset), global_step=itr)
        
        network.eval()
        total_log_p_x = 0.0
        total_elbo = 0.0
        for idx, img in enumerate(test_loader):
            img = img.to(device)
            z, log_q_z = variational(img)
            log_p_xz = network(z,img)
            elbo = log_p_xz - log_q_z
            log_p_x = torch.logsumexp(elbo, dim=1)
            total_elbo += elbo.cpu().numpy().sum()
            total_log_p_x += log_p_x.cpu().numpy().sum()
        total_elbo /= len(test_loader.dataset)
        total_log_p_x /= len(test_loader.dataset)
        valid_logger.add_scalar('elbo', total_elbo, global_step=itr)
        valid_logger.add_scalar('log_p_x', total_log_p_x, global_step=itr)