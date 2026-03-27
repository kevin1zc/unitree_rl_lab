from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.distributions import Normal


def _init_linear(linear: nn.Linear, init_dist: str) -> None:
    fan_in = linear.in_features
    if init_dist == "uniform":
        bound = math.sqrt(3.0) / math.sqrt(float(fan_in))
        nn.init.uniform_(linear.weight, -bound, bound)
    elif init_dist == "normal":
        std = 1.0 / math.sqrt(float(fan_in))
        nn.init.normal_(linear.weight, mean=0.0, std=std)
    elif init_dist == "orthogonal":
        nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
    else:
        raise ValueError(f"Unknown init_dist: {init_dist}")
    nn.init.zeros_(linear.bias)


def _resolve_activation(activation: str | type[nn.Module]) -> type[nn.Module]:
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation
    if not isinstance(activation, str):
        raise TypeError(f"Unsupported activation spec: {activation!r}")
    name = activation.lower()
    if name == "tanh":
        return nn.Tanh
    if name == "elu":
        return nn.ELU
    if name == "relu":
        return nn.ReLU
    raise ValueError(f"Unknown activation: {activation}")


class RandomFeatureMLP(nn.Module):
    """Randomized feature network with frozen hidden layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        activation: str | type[nn.Module] = nn.Tanh,
        init_dist: str = "uniform",
    ) -> None:
        super().__init__()
        activation_cls = _resolve_activation(activation)
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            linear = nn.Linear(dims[index], dims[index + 1])
            _init_linear(linear, init_dist=init_dist)
            layers.append(linear)
            if index < len(dims) - 2:
                layers.append(activation_cls())
        self.net = nn.Sequential(*layers)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class RandomFeatureValue(nn.Module):
    """V(s) with frozen random features and a trainable linear head."""

    def __init__(
        self,
        obs_dim: int,
        feature_dim: int,
        hidden_dims: Iterable[int],
        activation: str | type[nn.Module] = nn.Tanh,
        init_dist: str = "uniform",
    ) -> None:
        super().__init__()
        self.trunk = RandomFeatureMLP(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=feature_dim,
            activation=activation,
            init_dist=init_dist,
        )
        self.head = nn.Linear(feature_dim, 1)
        if init_dist == "orthogonal":
            nn.init.orthogonal_(self.head.weight, gain=1.0)
        else:
            nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.trunk(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(obs)
        return self.head(feats).squeeze(-1)

    @torch.no_grad()
    def set_head(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        self.head.weight.copy_(weight.view_as(self.head.weight))
        self.head.bias.copy_(bias.view_as(self.head.bias))


class RandomFeatureGaussianPolicy(nn.Module):
    """Gaussian policy with frozen random features and trainable mean/log-std."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_dim: int,
        hidden_dims: Iterable[int],
        activation: str | type[nn.Module] = nn.Tanh,
        init_dist: str = "uniform",
        init_log_std: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.trunk = RandomFeatureMLP(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=feature_dim,
            activation=activation,
            init_dist=init_dist,
        )
        self.head = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        if init_dist == "orthogonal":
            nn.init.orthogonal_(self.head.weight, gain=0.01)
        else:
            nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.trunk(obs)

    def distribution_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.extract_features(obs)
        mean = self.head(feats)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        return mean, std

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean, std = self.distribution_params(obs)
        return Normal(mean, std, validate_args=False)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self._distribution(obs).mean

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy

    def top_parameters(self):
        return list(self.head.parameters()) + [self.log_std]


def fit_ridge_closed_form(
    features: torch.Tensor, targets: torch.Tensor, l2_reg: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve linear ridge regression with an explicit bias term."""
    device = features.device
    num_samples, feature_dim = features.shape
    dtype = torch.float32

    x = torch.cat(
        [features.to(dtype=dtype), torch.ones(num_samples, 1, device=device, dtype=dtype)],
        dim=1,
    )
    eye = torch.eye(feature_dim + 1, device=device, dtype=dtype)
    eye[-1, -1] = 0.0
    a = x.T @ x + float(l2_reg) * eye
    b = x.T @ targets.to(dtype=dtype)
    beta = torch.linalg.solve(a, b)
    weight = beta[:-1].to(dtype=torch.float32).reshape(1, -1)
    bias = beta[-1:].to(dtype=torch.float32).reshape(1)
    return weight, bias
