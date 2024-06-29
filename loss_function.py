import torch

def nt_xent_loss(queries, keys, temperature=0.1):
    b, device = queries.shape[0], queries.device
    n = b * 2
    projs = torch.cat((queries, keys))
    logits = projs @ projs.t()

    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)
    logits /= temperature

    labels = torch.cat([torch.arange(b, device=device) + b - 1, torch.arange(b, device=device)])
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction='sum')
    loss /= n
    return loss
