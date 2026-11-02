import unittest

import torch

from models.MICN import Model as MICN


def _configs(**overrides):
    configs = {
        'd_model': 16,
        'n_heads': 2,
        'd_layers': 1,
        'dropout': 0.05,
        'embed': 'fixed',
        'freq': 'h',
        'conv_kernel': [12, 16],
        'cat_emb_dim': 2,
        'METADATA': {
            'seq_len': 120,
            'pred_len': 60,
            'feature_types': {'n_cat': 2, 'n_num': 3, 'n_target': 1},
            'input_numerical_features': ['a', 'target', 'b'],
            'target_features': ['target'],
        },
    }
    configs.update(overrides)
    return configs


class MICNOptionsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.input_cat = torch.randint(0, 3, (4, 120, 2)).float()
        self.input_num = torch.randn(4, 120, 3)
        self.shift = torch.tensor([5.0, -3.0, 2.0])  # per numerical channel; target is index 1

    def _predict(self, model, input_num):
        model.eval()
        with torch.no_grad():
            return model(input_cat=self.input_cat, input_num=input_num)

    def test_config_conv_kernel_and_dropout_are_used(self):
        model = MICN(_configs(conv_kernel=[10, 20, 30], dropout=0.2))
        mic = model.conv_trans.mic[0]
        self.assertEqual([conv.kernel_size[0] for conv in mic.conv], [10, 20, 30])
        self.assertAlmostEqual(mic.drop.p, 0.2)
        self.assertEqual(self._predict(model, self.input_num).shape, (4, 60, 1))

    def test_default_keeps_previous_behaviour(self):
        model = MICN(_configs())
        self.assertFalse(model.use_norm)
        self.assertEqual([conv.kernel_size[0] for conv in model.conv_trans.mic[0].conv], [12, 16])
        base = self._predict(model, self.input_num)
        shifted = self._predict(model, self.input_num + self.shift)
        # Without instance normalization a level shift is not carried through exactly.
        self.assertFalse(torch.allclose(shifted, base + self.shift[1], atol=1e-4))

    def test_use_norm_follows_target_level_shift(self):
        model = MICN(_configs(use_norm=True))
        self.assertEqual(model.target_channel_indices, (2 * 2 + 1,))
        base = self._predict(model, self.input_num)
        shifted = self._predict(model, self.input_num + self.shift)
        self.assertEqual(base.shape, (4, 60, 1))
        # Shifting every numerical channel by a constant leaves the normalized input
        # unchanged, so the forecast moves by exactly the target channel's shift.
        torch.testing.assert_close(shifted, base + self.shift[1], atol=1e-4, rtol=0)


if __name__ == '__main__':
    unittest.main()
