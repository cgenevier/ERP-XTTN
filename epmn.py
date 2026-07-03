"""ERP Prototypical Matching Net (EPMN) — Wei et al. (2022).

Faithful reimplementation of the metric-based meta-learning zero-calibration
classifier (no official code is published). The feature extractor follows the
Manor-CNN architecture of the paper's Table 2, adapted only in input
dimensionality (n_channels x T) — no internal time resampling — so the input
is identical to the other models. Prototype construction, the squared-Euclidean
distance metric, the softmax attention kernel, and the classification + metric
losses follow the paper's Eqs. (1)-(8); the episodic meta-training strategy
(Algorithm 1) is implemented in ``04_train.py`` (run_fold_epmn), wrapped in the
shared two-phase / early-stopping protocol.

Reference:
    Wei, Qiu, Zhang, Mao, He, "ERP prototypical matching net: a meta-learning
    method for zero-calibration RSVP-based image retrieval,"
    J. Neural Eng. 19(2):026028, 2022.

Shapes:
    input EEG sample  x   : (B, C, T)
    embedding          f(x): (B, embed_dim)  L2-normalized
    prototypes         p   : (2, embed_dim)  one per class {0, 1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EPMN(nn.Module):
    """EPMN feature extractor f (paper Table 2), adapted to (C, T) inputs.

    Three convolutional stages (spatial -> temporal -> temporal), each with
    batch-norm and ReLU, two intervening temporal max-pools, then a linear
    map to ``embed_dim`` followed by L2 normalization.
    """

    def __init__(self, n_channels: int, n_times: int, embed_dim: int = 1024):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.embed_dim = embed_dim

        # Stage 1: spatial convolution across all channels (kernel (C, 1))
        self.conv1 = nn.Conv2d(1, 96, kernel_size=(n_channels, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        # Stage 2: temporal convolution (kernel (1, 6))
        self.conv2 = nn.Conv2d(96, 128, kernel_size=(1, 6), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        # Stage 3: temporal convolution (kernel (1, 6))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(1, 6), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()

        # Flatten size computed via a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat = self._conv_forward(dummy).flatten(1)
            flat_size = flat.shape[1]

        self.fc = nn.Linear(flat_size, embed_dim)

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self._conv_forward(x).flatten(1)
        x = self.fc(x)
        # L2 normalize (paper Table 2, layer 13)
        return F.normalize(x, p=2, dim=1)


def class_template(X: torch.Tensor, y: torch.Tensor, k: int):
    """Mean waveform (ERP template) over class-k trials, Eq. (1).

    Args:
        X: (N, C, T) trials for one subject
        y: (N,) labels
    Returns:
        (C, T) template, or None if the subject has no class-k trials.
    """
    mask = (y == k)
    if mask.sum() == 0:
        return None
    return X[mask].mean(dim=0)


def build_prototypes(model: EPMN, support_templates, device) -> torch.Tensor:
    """Class prototypes = mean over support subjects of embedded ERP templates.

    All templates of a class are embedded in a single batch so BatchNorm sees
    a real batch rather than size-1 inputs.

    Args:
        model: EPMN feature extractor
        support_templates: list per subject of {k: (C, T) template tensor}
        device: torch device
    Returns:
        (2, embed_dim) prototype tensor. Gradients flow through ``model``.
    """
    protos = []
    for k in (0, 1):
        tmpls = [tmpl[k] for tmpl in support_templates
                 if k in tmpl and tmpl[k] is not None]
        if not tmpls:
            raise RuntimeError(f"No support subject has class {k}; cannot build prototype.")
        batch = torch.stack(tmpls, dim=0).to(device)   # (n_k, C, T)
        embs = model(batch)                            # (n_k, D)
        protos.append(embs.mean(dim=0, keepdim=True))  # (1, D)
    return torch.cat(protos, dim=0)                    # (2, D)


def squared_distances(emb: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance to each prototype, Eq. (3).

    Args:
        emb:    (B, D)
        protos: (2, D)
    Returns:
        (B, 2) distances.
    """
    return torch.cdist(emb, protos, p=2) ** 2


def epmn_class_logits(distances: torch.Tensor) -> torch.Tensor:
    """Class logits = -distance, so softmax(logits) = softmax(-d), Eqs. (4)-(5).

    p(y=k|x) = softmax(-d)_k (smaller distance -> higher probability).
    """
    return -distances


def epmn_metric_loss(distances: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Metric-learning loss, Eq. (7): pull to own prototype, push from other.

    mean_i [ d(x_i, p^{y_i}) - d(x_i, p^{1 - y_i}) ]
    """
    idx = torch.arange(distances.shape[0], device=distances.device)
    d_own = distances[idx, y]
    d_other = distances[idx, 1 - y]
    return (d_own - d_other).mean()
