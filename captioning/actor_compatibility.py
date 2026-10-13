"""Actor/action compatibility from labelled actor contrasts, no union input."""
import torch
from torch import nn
from torch.nn import functional as F
from captioning.supervised_relations import contrast_edges


def actor_supervision(rows, exhaustive=False):
    """Only endpoints of confirmed same-object actor edges are supervised."""
    labels = torch.full((len(rows), 6), -1, dtype=torch.long)
    edges = [e for e in contrast_edges(rows, max_per_positive=len(rows) if exhaustive else 4)
             if e[-1] == 'actor']
    for i, r, j, c, _ in edges:
        if rows[i]['labels'][r] != 1 or rows[j]['labels'][c] != 0:
            raise ValueError('Actor edge must have confirmed labels.')
        labels[i, r], labels[j, c] = 1, 0
    return labels, torch.tensor([e[:4] for e in edges], dtype=torch.long).reshape(-1, 4)


def actor_edge_metrics(scores, rows, exhaustive=False):
    _,edges=actor_supervision(rows,exhaustive=exhaustive)
    if not len(edges):return {'edges':0,'wins':0,'ties':0,'win_rate':None}
    i,r,j,c=edges.T
    gap=scores[i,r]-scores[j,c]
    return {'edges':len(edges),'images':len({rows[int(e[0])]['image_id'] for e in edges}),
            'wins':int((gap>0).sum()),'ties':int((gap==0).sum()),
            'win_rate':float((gap>0).float().mean()),'mean_gap':float(gap.mean())}


def pair_geometry(rows):
    # Absolute boxes plus offsets/area ratio: continuous evidence, not a distance rule.
    values = []
    for row in rows:
        a, b = row['subject_box'], row['object_box']
        if any(len(box) != 4 or not 0 <= box[0] < box[2] <= 1
               or not 0 <= box[1] < box[3] <= 1 for box in (a, b)):
            raise ValueError('Expected normalized nonempty boxes.')
        ac = ((a[0]+a[2])/2, (a[1]+a[3])/2)
        bc = ((b[0]+b[2])/2, (b[1]+b[3])/2)
        values.append(list(a)+list(b)+[bc[0]-ac[0], bc[1]-ac[1]])
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 10)


class ActorCompatibility(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4*dim+10, 128), nn.GELU(),
                                     nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 6))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, views, geometry, text=None):
        a, b = views[:, 0].detach(), views[:, 1].detach()
        return self.network(torch.cat([a, b, a*b, a-b, geometry], dim=-1))


class CompactActorCompatibility(nn.Module):
    """150 parameters; fixed CLIP pair/text evidence plus normalized geometry."""
    def __init__(self):
        super().__init__()
        self.network = nn.Linear(24, 6)
        nn.init.zeros_(self.network.weight)
        nn.init.zeros_(self.network.bias)

    def forward(self, views, geometry, text):
        a, b = views[:, 0].detach(), views[:, 1].detach()
        target = F.normalize(text.detach(), dim=-1)
        semantic = torch.cat([torch.einsum('bd,bkd->bk', a, target),
                              torch.einsum('bd,bkd->bk', b, target)], dim=-1)
        similarity = (a*b).sum(-1, keepdim=True)
        distance = geometry[:, 8:10].square().sum(-1, keepdim=True)
        return self.network(torch.cat([geometry, semantic, similarity, distance], dim=-1))


def actor_loss(logits, labels, edges):
    known = labels >= 0
    if not known.any() or not len(edges):
        raise ValueError('No confirmed actor supervision.')
    i, r, j, c = edges.T
    if not ((labels[i, r] == 1).all() and (labels[j, c] == 0).all()):
        raise ValueError('Invalid actor labels.')
    bce = F.binary_cross_entropy_with_logits(logits[known], labels[known].float())
    ranking = F.softplus(logits[j, c]-logits[i, r]).mean()
    return .5*bce + .5*ranking, bce, ranking
