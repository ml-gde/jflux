import numpy as np

from flux.modules.conditioner import HFEmbedder as TorchHFEmbedder
from jflux.modules.conditioner import HFEmbedder as JaxHFEmbedder


class HFEmbedderTestCase(np.testing.TestCase):
    def test_hf_embed(self):
        # initialize layers
        TorchHFEmbedder()