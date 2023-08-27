
import re
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from compressai.models.base import CompressionModel
from compressai.models.utils import conv, deconv
# from compressai.registry import register_model


# @register_model("bmshj2018-factorized")
class CustomFactorizedPrior(CompressionModel):

    def __init__(self, N, M, L, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(M)

        if L < 1:
            raise ValueError("L must be an integer greater than 0.")

        if L == 1:
            self.g_a = nn.Sequential(conv(3, M))
        elif L > 1:
            modules_g_a = []
            for i in range(L):
                if i == 0:
                    modules_g_a.append(conv(3, N))
                    modules_g_a.append(GDN(N))
                elif i > 0 and i < L-1:
                    modules_g_a.append(conv(N, N))
                    modules_g_a.append(GDN(N))
                elif i == L-1:
                    modules_g_a.append(conv(N, M))
            self.g_a = nn.Sequential(*modules_g_a)

        if L == 1:
            self.g_s = nn.Sequential(deconv(M, 3))
        elif L > 1:
            modules_g_s = []
            for i in range(L):
                if i == 0:
                    modules_g_s.append(deconv(M, N))
                    modules_g_s.append(GDN(N, inverse=True))
                elif i > 0 and i < L-1:
                    modules_g_s.append(deconv(N, N))
                    modules_g_s.append(GDN(N, inverse=True))
                elif i == L-1:
                    modules_g_s.append(deconv(N, 3))
            self.g_s = nn.Sequential(*modules_g_s)

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2**4

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_s.0.weight"].size(0)
        compiled_regex = re.compile('g_a\.[\d]+\.weight')
        L = len(list(filter(compiled_regex.search,state_dict.keys())))
        net = cls(N, M, L)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}