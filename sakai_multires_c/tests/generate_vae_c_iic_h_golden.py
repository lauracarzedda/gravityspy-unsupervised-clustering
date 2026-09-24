"""Independent PyTorch EVAL golden for MINI75-IIC-6B (synthetic only).

Run in an environment with PyTorch. This script reads no project artifacts,
uses no Gravity Spy labels, and creates no files. Compare GOLDEN_VALUES against
`test_vae_c_iic_h --golden-dump` with float32 tolerance after running both.
"""

import json
import torch


ROWS = 4
INPUT_DIM = 3
HIDDEN = 512
C = 36
K = 5


def main_head(k: int) -> torch.nn.Module:
    model = torch.nn.Sequential(
        torch.nn.Linear(INPUT_DIM, HIDDEN, bias=False),
        torch.nn.BatchNorm1d(HIDDEN, eps=1e-5, momentum=0.1),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, C, bias=True),
        torch.nn.Softmax(dim=1),
    )
    with torch.no_grad():
        w1 = [
            0.006 * (((hidden * 3 + d * 7 + k) % 17) - 8)
            for hidden in range(HIDDEN)
            for d in range(INPUT_DIM)
        ]
        gamma = [1.0 + 0.001 * ((hidden % 7) - 3) for hidden in range(HIDDEN)]
        beta = [0.01 * ((hidden % 5) - 2) for hidden in range(HIDDEN)]
        mean = [0.002 * ((hidden % 13) - 6) for hidden in range(HIDDEN)]
        variance = [1.0 + 0.001 * (hidden % 17) for hidden in range(HIDDEN)]
        w2 = [
            0.0002 * (((c * 7 + hidden * 3 + k) % 19) - 9)
            for c in range(C)
            for hidden in range(HIDDEN)
        ]
        bias2 = [0.003 * (c - 17) + 0.005 * k for c in range(C)]
        model[0].weight.copy_(torch.tensor(w1, dtype=torch.float32).reshape(HIDDEN, INPUT_DIM))
        model[1].weight.copy_(torch.tensor(gamma, dtype=torch.float32))
        model[1].bias.copy_(torch.tensor(beta, dtype=torch.float32))
        model[1].running_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        model[1].running_var.copy_(torch.tensor(variance, dtype=torch.float32))
        model[3].weight.copy_(torch.tensor(w2, dtype=torch.float32).reshape(C, HIDDEN))
        model[3].bias.copy_(torch.tensor(bias2, dtype=torch.float32))
    model.eval()
    return model


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    mu = torch.tensor(
        [[0.1 * ((row + 1) * (d + 1)) - 0.25 for d in range(INPUT_DIM)]
         for row in range(ROWS)],
        dtype=torch.float32,
    )
    models = [main_head(k) for k in range(K)]
    bn_before = [
        (m[1].running_mean.clone(), m[1].running_var.clone(),
         m[1].num_batches_tracked.clone())
        for m in models
    ]
    with torch.inference_mode():
        blocks = [m(mu) for m in models]
        h = torch.cat(blocks, dim=1).contiguous()
    assert h.shape == (ROWS, K * C)
    assert h.dtype == torch.float32
    assert torch.isfinite(h).all()
    assert all(torch.allclose(b.sum(dim=1), torch.ones(ROWS), atol=1e-4, rtol=0)
               for b in blocks)
    assert all(
        torch.equal(m[1].running_mean, state[0])
        and torch.equal(m[1].running_var, state[1])
        and torch.equal(m[1].num_batches_tracked, state[2])
        for m, state in zip(models, bn_before)
    )
    print("PYTORCH_VERSION=" + torch.__version__)
    print("GOLDEN_SHAPE=4x180")
    print("GOLDEN_VALUES=" + json.dumps(h.flatten().tolist(), separators=(",", ":")))
    print("BN_EVAL_UNCHANGED=true")


if __name__ == "__main__":
    main()
