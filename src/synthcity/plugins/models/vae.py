# stdlib
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# synthcity absolute
import synthcity.logger as log

# synthcity relative
from .mlp import MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_in: int,
        n_units_embedding: int,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        seed: int = 0,
        dropout: float = 0.1,
        batch_norm: bool = True,
        residual: bool = False,
    ) -> None:
        super(Encoder, self).__init__()
        self.model = MLP(
            task_type="regression",
            n_units_in=n_units_in,
            n_units_out=n_units_hidden,
            n_units_hidden=n_units_hidden,
            n_layers_hidden=n_layers_hidden - 1,
            nonlin=nonlin,
            seed=seed,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
        ).to(DEVICE)

        self.mu_fc = nn.Linear(n_units_hidden, n_units_embedding).to(DEVICE)
        self.logvar_fc = nn.Linear(n_units_hidden, n_units_embedding).to(DEVICE)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        shared = self.model(X)
        mu = self.mu_fc(shared)
        logvar = self.mu_fc(shared)
        return mu, logvar


class Decoder(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_embedding: int,
        n_units_out: int,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        nonlin_out: Optional[List[Tuple[str, int]]] = None,
        seed: int = 0,
        dropout: float = 0.1,
        batch_norm: bool = True,
        residual: bool = False,
    ) -> None:
        super(Decoder, self).__init__()
        self.model = MLP(
            task_type="regression",
            n_units_in=n_units_embedding,
            n_units_out=n_units_out,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            nonlin=nonlin,
            nonlin_out=nonlin_out,
            seed=seed,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
        ).to(DEVICE)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, X: Tensor) -> Tensor:
        return self.model(X)


class VAE(nn.Module):
    """Basic VAE implementation."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_features: int,
        n_units_embedding: int,
        batch_size: int = 100,
        n_iter: int = 500,
        n_iter_print: int = 10,
        seed: int = 0,
        clipping_value: int = 1,
        lr: float = 2e-4,
        weight_decay: float = 1e-3,
        loss_factor: int = 2,
        decoder_n_layers_hidden: int = 2,
        decoder_n_units_hidden: int = 250,
        decoder_nonlin: str = "leaky_relu",
        decoder_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        decoder_batch_norm: bool = False,
        decoder_dropout: float = 0,
        decoder_residual: bool = True,
        encoder_n_layers_hidden: int = 3,
        encoder_n_units_hidden: int = 300,
        encoder_nonlin: str = "leaky_relu",
        encoder_batch_norm: bool = False,
        encoder_dropout: float = 0.1,
    ) -> None:
        super(VAE, self).__init__()

        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.clipping_value = clipping_value
        self.loss_factor = loss_factor
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_units_embedding = n_units_embedding

        self.seed = seed
        torch.manual_seed(self.seed)

        self.encoder = Encoder(
            n_features,
            n_units_embedding,
            n_layers_hidden=encoder_n_layers_hidden,
            n_units_hidden=encoder_n_units_hidden,
            nonlin=encoder_nonlin,
            batch_norm=encoder_batch_norm,
            dropout=encoder_dropout,
        )
        self.decoder = Decoder(
            n_units_embedding,
            n_features,
            n_layers_hidden=decoder_n_layers_hidden,
            n_units_hidden=decoder_n_units_hidden,
            nonlin=decoder_nonlin,
            nonlin_out=decoder_nonlin_out,
            batch_norm=decoder_batch_norm,
            dropout=decoder_dropout,
            residual=decoder_residual,
        )

        if decoder_nonlin_out is None:
            decoder_nonlin_out = [("none", n_features)]
        self.decoder_nonlin_out = decoder_nonlin_out

    def fit(self, X: np.ndarray) -> Any:
        Xt = self._check_tensor(X)

        self._train(Xt)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(self, samples: int) -> np.ndarray:
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.n_units_embedding)
            std = torch.ones(self.batch_size, self.n_units_embedding)

            noise = torch.normal(mean=mean, std=std).to(DEVICE)
            fake = self.decoder(noise)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return data

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _train(self, X: Tensor) -> Any:
        loader = self._dataloader(X)

        optimizer = Adam(
            self.parameters(),
            weight_decay=self.weight_decay,
            lr=self.lr,
        )

        for epoch in range(self.n_iter):
            for id_, data in enumerate(loader):
                optimizer.zero_grad()

                real = data[0].to(DEVICE)

                mu, logvar = self.encoder(real)
                embedding = self._reparameterize(mu, logvar)
                reconstructed = self.decoder(embedding)
                loss = self._loss_function(
                    reconstructed,
                    real,
                    mu,
                    logvar,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

                optimizer.step()
            if epoch % self.n_iter_print == 0:
                log.info(f"[{epoch}/{self.n_iter}] Loss: {loss.detach()}")

    def _check_tensor(self, X: Tensor) -> Tensor:
        if isinstance(X, Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)

    def _dataloader(self, X: Tensor) -> DataLoader:
        dataset = TensorDataset(X)
        return DataLoader(dataset, batch_size=self.batch_size, pin_memory=False)

    def _loss_function(
        self,
        reconstructed: Tensor,
        real: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        step = 0

        loss = []
        for activation, length in self.decoder_nonlin_out:
            step_end = step + length
            if activation == "softmax":
                discr_loss = nn.functional.cross_entropy(
                    reconstructed[:, step:step_end],
                    torch.argmax(real[:, step:step_end], dim=-1),
                    reduction="sum",
                )
                loss.append(discr_loss)
            else:
                cont_loss = (
                    reconstructed[:, step:step_end] - real[:, step:step_end]
                ) ** 2
                cont_loss = torch.sum(cont_loss)
                loss.append(cont_loss)
            step = step_end

        if step != reconstructed.size()[1]:
            raise RuntimeError(
                f"Invalid reconstructed features. Expected {step}, got {reconstructed.shape}"
            )

        reconstruction_loss = torch.sum(torch.FloatTensor(loss))

        KLD_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()))

        return reconstruction_loss * self.loss_factor + KLD_loss