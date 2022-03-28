"""Anonymization through Data Synthesis using Generative Adversarial Networks:
A harmonizing advancement for AI in medicine (ADS-GAN) Codebase.

Reference: Jinsung Yoon, Lydia N. Drumright, Mihaela van der Schaar,
"Anonymization through Data Synthesis using Generative Adversarial Networks (ADS-GAN):
A harmonizing advancement for AI in medicine,"
IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
Paper link: https://ieeexplore.ieee.org/document/9034117
"""
# stdlib
from typing import Any, List

# third party
import pandas as pd

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.models import TabularVAE


class RTVAEPlugin(Plugin):
    """RTVAE plugin.

    Args:
        decoder_n_layers_hidden: int
            Number of hidden layers in the decoder
        decoder_n_units_hidden: int
            Number of hidden units in each layer of the decoder
        decoder_nonlin: string, default 'tanh'
            Nonlinearity to use in the decoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        decoder_dropout: float
            Dropout value. If 0, the dropout is not used.
        encoder_n_layers_hidden: int
            Number of hidden layers in the encoder
        encoder_n_units_hidden: int
            Number of hidden units in each layer of the encoder
        encoder_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the encoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        encoder_dropout: float
            Dropout value for the encoder. If 0, the dropout is not used.
        n_iter: int
            Maximum number of iterations in the encoder.
        lr: float
            learning rate for optimizer. step_size equivalent in the JAX version.
        weight_decay: float
            l2 (ridge) penalty for the weights.
        batch_size: int
            Batch size
        seed: int
            Seed used
        clipping_value: int, default 1
            Gradients clipping value
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("adsgan")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_iter: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 64,
        seed: int = 0,
        clipping_value: int = 1,
        encoder_max_clusters: int = 20,
        decoder_n_layers_hidden: int = 2,
        decoder_n_units_hidden: int = 100,
        decoder_nonlin: str = "tanh",
        decoder_dropout: float = 0,
        encoder_n_layers_hidden: int = 2,
        encoder_n_units_hidden: int = 100,
        encoder_nonlin: str = "leaky_relu",
        encoder_dropout: float = 0.1,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.decoder_n_layers_hidden = decoder_n_layers_hidden
        self.decoder_n_units_hidden = decoder_n_units_hidden
        self.decoder_nonlin = decoder_nonlin
        self.decoder_dropout = decoder_dropout
        self.encoder_n_layers_hidden = encoder_n_layers_hidden
        self.encoder_n_units_hidden = encoder_n_units_hidden
        self.encoder_nonlin = encoder_nonlin
        self.encoder_dropout = encoder_dropout
        self.n_iter = n_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed
        self.clipping_value = clipping_value
        self.encoder_max_clusters = encoder_max_clusters

    @staticmethod
    def name() -> str:
        return "rtvae"

    @staticmethod
    def type() -> str:
        return "vae"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=100, high=500, step=100),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            IntegerDistribution(name="decoder_n_layers_hidden", low=1, high=5),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[64, 128, 256, 512]),
            IntegerDistribution(
                name="decoder_n_units_hidden", low=50, high=500, step=50
            ),
            CategoricalDistribution(
                name="decoder_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            FloatDistribution(name="decoder_dropout", low=0, high=0.2),
            IntegerDistribution(name="encoder_n_layers_hidden", low=1, high=5),
            IntegerDistribution(
                name="encoder_n_units_hidden", low=50, high=500, step=50
            ),
            CategoricalDistribution(
                name="encoder_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            FloatDistribution(name="encoder_dropout", low=0, high=0.2),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "RTVAEPlugin":
        features = X.shape[1]
        self.model = TabularVAE(
            X,
            n_units_embedding=features,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            n_iter=self.n_iter,
            decoder_n_layers_hidden=self.decoder_n_layers_hidden,
            decoder_n_units_hidden=self.decoder_n_units_hidden,
            decoder_nonlin=self.decoder_nonlin,
            decoder_nonlin_out_discrete="softmax",
            decoder_nonlin_out_continuous="tanh",
            decoder_residual=True,
            decoder_batch_norm=False,
            decoder_dropout=0,
            encoder_n_units_hidden=self.encoder_n_units_hidden,
            encoder_n_layers_hidden=self.encoder_n_layers_hidden,
            encoder_nonlin=self.encoder_nonlin,
            encoder_batch_norm=False,
            encoder_dropout=self.encoder_dropout,
            clipping_value=self.clipping_value,
            encoder_max_clusters=self.encoder_max_clusters,
        )
        self.model.fit(X)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.generate, count, syn_schema)


plugin = RTVAEPlugin