import numpy as np
import torch
import tqdm

sigma_min=0.01 * np.pi
sigma_max=np.pi
num_nodes = 100
edge = 12
sigma = np.exp(np.random.uniform(low=np.log(sigma_min), high=np.log(sigma_max)))  # [0, pi]
node_sigma = sigma * torch.ones(num_nodes)
torsion_updates = np.random.normal(loc=0.0, scale=sigma, size=edge) # mean 0 std[0, pi]
data_pos = transform_structure(pos, torsion_updates)

pred = model(data_pos)

node_sigma = torch.log(node_sigma / sigma_min) / np.log(sigma_max / sigma_min) * 10000
node_sigma_emb = get_timestep_embedding(node_sigma, self.sigma_embed_dim)

######################################################################################################
X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

def grad(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):# 为什么需要这个for loop 啊 还是一个累加的关系
        p_ += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_

def p(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_

x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi # [0, pi]
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi # [0, 2pi]

p_ = p(x, sigma[:, None], N=100)
score_ = grad(x, sigma[:, None], N=100) / p_

def score(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi #[-pi pi]
    sign = np.sign(x)
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_[sigma, x]

def sample(sigma):
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out

score_norm_ = score(
    sample(sigma[None].repeat(10000, 0).flatten()),
    sigma[None].repeat(10000, 0).flatten()
).reshape(10000, -1)

def score_norm(sigma):
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return score_norm_[sigma]

######################################################################################################

score = score(torsion_updates, sigma)
score_norm = score_norm(sigma)
# predict the gradient directly rather than delta_tal
loss = ((score - pred) ** 2 / score_norm).mean()

def sample(conformers, model, sigma_max=np.pi, sigma_min=0.01 * np.pi, steps=20, batch_size=32,
           ode=False, likelihood=None, pdb=None):
    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(conf_dataset, batch_size=batch_size, shuffle=False)

    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1]
    eps = 1 / steps

    for batch_idx, data in enumerate(loader):

        dlogp = torch.zeros(data.num_graphs)
        data_gpu = copy.deepcopy(data).to(device)
        for sigma_idx, sigma in enumerate(sigma_schedule):

            data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
            with torch.no_grad():
                data_gpu = model(data_gpu)

            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape)
            score = data_gpu.edge_pred.cpu()


            perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

            conf_dataset.apply_torsion_and_update_pos(data, perturb.numpy())
            data_gpu.pos = data.pos.to(device)

            if pdb:
                for conf_idx in range(data.num_graphs):
                    coords = data.pos[data.ptr[conf_idx]:data.ptr[conf_idx + 1]]
                    num_frames = still_frames if sigma_idx == steps - 1 else 1
                    pdb.add(coords, part=batch_size * batch_idx + conf_idx, order=sigma_idx + 2, repeat=num_frames)

            for i, d in enumerate(dlogp):
                conformers[data.idx[i]].dlogp = d.item()

    return conformers