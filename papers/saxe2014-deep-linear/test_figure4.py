"""Check the covariance update and active-subspace reduction against autograd."""

import unittest

import torch

from figure4 import one_step


class Figure4Checks(unittest.TestCase):
    def test_one_step_matches_full_width_autograd(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        x = torch.rand(12, 5)
        y = torch.rand(12, 2)
        xx, yx = x.T @ x, y.T @ x
        w0 = torch.rand(2, 5) * 0.1
        w1 = torch.rand(2, 2) * 0.1
        w2 = torch.rand(2, 2) * 0.1

        # Embed the two active channels in a four-unit hidden layer. The
        # remaining two channels have orthogonal identity propagation.
        full0 = torch.cat((w0, torch.zeros(2, 5)), dim=0).requires_grad_()
        full1 = torch.block_diag(w1, torch.eye(2)).requires_grad_()
        full2 = torch.cat((w2, torch.zeros(2, 2)), dim=1).requires_grad_()
        prediction = (full2 @ full1 @ full0 @ x.T).T
        loss = 0.5 * ((y - prediction) ** 2).sum()
        loss.backward()
        step = 1e-3
        updated_full = [w.detach() - step * w.grad for w in (full0, full1, full2)]

        reduced = [w.unsqueeze(0).clone() for w in (w0, w1, w2)]
        error = one_step(reduced, xx, yx, float((y * y).sum()),
                         torch.tensor([step]), torch.tensor([True]))
        self.assertAlmostEqual(float(error[0]), float(loss), places=5)
        full_map = updated_full[2] @ updated_full[1] @ updated_full[0]
        reduced_map = reduced[2][0] @ reduced[1][0] @ reduced[0][0]
        self.assertLess(float((full_map - reduced_map).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
