"""LSTM, dilated 1D-CNN and PatchTST with a quantile head (24 h x 2 targets x 3 quantiles = 72 outputs).

Every model takes the 168 h history x of shape (B, 168, C) and optionally the known-future inputs f of the
target day, shape (B, 24, K) (calendar, official day-ahead forecasts). f is flattened and concatenated to
the features that feed the output head."""
import torch
import torch.nn as nn

QUANTILES = (0.1, 0.5, 0.9)
OUT = 24 * 2 * len(QUANTILES)


def pinball_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """pred (B, 24, 2, 3), y (B, 24, 2). Mean over everything."""
    q = torch.tensor(QUANTILES, device=pred.device, dtype=pred.dtype)
    pred = torch.sort(pred, dim=-1).values  # same as evaluate.metrics: no crossing quantiles
    d = y.unsqueeze(-1) - pred
    return torch.maximum(q * d, (q - 1) * d).mean()


def _cat(z: torch.Tensor, f):
    return z if f is None or f.shape[-1] == 0 else torch.cat([z, f.flatten(1)], dim=1)


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, n_future: int = 0, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden + 24 * n_future, OUT)

    def forward(self, x, f=None):              # x (B, 168, C), f (B, 24, K)
        out, _ = self.lstm(x)
        return self.head(_cat(out[:, -1], f)).view(-1, 24, 2, len(QUANTILES))


CNN_DILATIONS = (1, 2, 4, 8, 16, 32)


def receptive_field(kernel: int, dilations=CNN_DILATIONS) -> int:
    """Hours seen by the last step of a stack of causal dilated convolutions."""
    return 1 + (kernel - 1) * sum(dilations)


class CNNForecaster(nn.Module):
    """6 causal dilated conv layers (dilation 1..32, kernel 4 -> receptive field 190 h >= 168 h, so the last
    step sees the whole week) -> last step + global average -> dense."""
    def __init__(self, n_features: int, n_future: int = 0, filters: int = 64, kernel: int = 4, dropout: float = 0.1):
        super().__init__()
        layers, c = [], n_features
        for d in CNN_DILATIONS:
            layers += [nn.ConstantPad1d(((kernel - 1) * d, 0), 0.0), nn.Conv1d(c, filters, kernel, dilation=d),
                       nn.ReLU(), nn.Dropout(dropout)]
            c = filters
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * filters + 24 * n_future, 128), nn.ReLU(), nn.Linear(128, OUT))

    def forward(self, x, f=None):              # x (B, 168, C), f (B, 24, K)
        h = self.conv(x.transpose(1, 2))       # (B, F, 168)
        z = torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1)
        return self.head(_cat(z, f)).view(-1, 24, 2, len(QUANTILES))


class PatchTSTForecaster(nn.Module):
    """PatchTST-style Transformer: the 168 h history is cut into overlapping patches (24 h, stride 12);
    each patch (all channels flattened) is one token; learned positional embedding; Transformer encoder
    (attention over patches); flattened tokens -> 72 quantile outputs.
    Note: the neuralforecast PatchTST is univariate (no exogenous inputs), so it is re-implemented here
    to allow the same multivariate inputs as the LSTM/CNN."""
    def __init__(self, n_features: int, n_future: int = 0, patch_len: int = 24, stride: int = 12, d_model: int = 128,
                 heads: int = 4, layers: int = 3, dropout: float = 0.2, lookback: int = 168):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        n_patches = (lookback - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len * n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_patches * d_model)
        self.head = nn.Sequential(nn.Linear(n_patches * d_model + 24 * n_future, 256),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(256, OUT))

    def forward(self, x, f=None):                           # x (B, 168, C), f (B, 24, K)
        p = x.unfold(1, self.patch_len, self.stride)        # (B, P, C, patch_len)
        t = self.drop(self.embed(p.flatten(2)) + self.pos)  # (B, P, d)
        z = self.encoder(t)
        return self.head(_cat(self.norm(z.flatten(1)), f)).view(-1, 24, 2, len(QUANTILES))


def build(name: str, n_features: int, n_future: int = 0, **hp) -> nn.Module:
    return {"lstm": LSTMForecaster, "cnn": CNNForecaster, "patchtst": PatchTSTForecaster}[name](n_features, n_future, **hp)
