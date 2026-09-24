"""Emit an independent CPU PyTorch golden for one MINI-C5B3 IIC pair.

Run this standalone where PyTorch is available (for example, Colab).  It
reads no project data and writes no files.  The fixture is intentionally
small, explicit, and unrelated to the official initializer or K=5 training.
The forward order is the literal Notebook06 order: main A, over A, main B,
over B.  Both views of each branch use the same parameters and BatchNorm.
"""

import json

import torch
import torch.nn as nn


BATCH = 4
INPUT_DIM = 3
HIDDEN = 512
MAIN_CLASSES = 3
OVER_CLASSES = 4
ACTIVE = 4


class IICHead(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(INPUT_DIM, HIDDEN, bias=False)
        self.bn = nn.BatchNorm1d(
            HIDDEN,
            eps=1.0e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.relu = nn.ReLU(inplace=False)
        self.output = nn.Linear(HIDDEN, output_dim, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.softmax(
            self.output(self.relu(self.bn(self.hidden(value)))),
            dim=1,
        )


class IICClassifierPair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main_head = IICHead(MAIN_CLASSES)
        self.overcluster_head = IICHead(OVER_CLASSES)

    def forward(self, value: torch.Tensor):
        # Python evaluates the dictionary values left-to-right.
        return {
            "main": self.main_head(value),
            "overcluster": self.overcluster_head(value),
        }


def initialize_common_tail(head: IICHead) -> None:
    head.hidden.weight.zero_()
    head.bn.weight.fill_(1.0)
    head.bn.bias.fill_(-1.0)
    head.bn.running_mean.zero_()
    head.bn.running_var.fill_(1.0)
    head.bn.num_batches_tracked.zero_()
    head.output.weight.zero_()
    head.output.bias.zero_()


def initialize_pair(pair: IICClassifierPair) -> None:
    with torch.no_grad():
        main = pair.main_head
        over = pair.overcluster_head
        initialize_common_tail(main)
        initialize_common_tail(over)

        # Reuse the explicit MINI-C5B2R main-head fixture, not its outputs.
        main.hidden.weight[:ACTIVE].copy_(torch.tensor([
            [0.5, -0.25, 0.125],
            [-0.375, 0.25, 0.5],
            [0.25, 0.75, -0.5],
            [-0.125, 0.5, 0.375],
        ]))
        main.bn.weight[:ACTIVE].copy_(torch.tensor([
            1.125, -0.875, 0.75, 1.25,
        ]))
        main.bn.bias[:ACTIVE].copy_(torch.tensor([
            0.125, 0.25, -0.0625, -0.125,
        ]))
        main.bn.running_mean[:ACTIVE].copy_(torch.tensor([
            0.25, -0.125, 0.375, -0.25,
        ]))
        main.bn.running_var[:ACTIVE].copy_(torch.tensor([
            1.5, 0.75, 2.0, 1.25,
        ]))
        main.output.weight[:, :ACTIVE].copy_(torch.tensor([
            [0.5, -0.25, 0.125, 0.0625],
            [-0.125, 0.375, 0.25, -0.5],
            [0.25, 0.125, -0.375, 0.625],
        ]))
        main.output.bias.copy_(torch.tensor([
            0.0625, -0.125, 0.1875,
        ]))

        # Arithmetic, dyadic values keep the over-head fixture easy to
        # reproduce in a C test without relying on PyTorch initialization.
        for h in range(ACTIVE):
            for d in range(INPUT_DIM):
                over.hidden.weight[h, d] = (((3 * h + d) % 7) - 3) * 0.125
            over.bn.weight[h] = 1.0 + ((h % 3) - 1) * 0.125
            over.bn.bias[h] = (((2 * h) % 5) - 2) * 0.125
            over.bn.running_mean[h] = (((h + 1) % 4) - 1) * 0.125
            over.bn.running_var[h] = 1.0 + (h + 1) * 0.25
        for o in range(OVER_CLASSES):
            for h in range(ACTIVE):
                over.output.weight[o, h] = (
                    (((5 * o + 2 * h) % 9) - 4) * 0.125
                )
            over.output.bias[o] = ((o % 3) - 1) * 0.0625


def negative_mutual_information(
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
):
    """Notebook06/C5B1: no epsilon, symmetry, or joint renormalization."""
    joint = probabilities_a.transpose(0, 1) @ probabilities_b / BATCH
    marginal_i = joint.sum(dim=1)
    marginal_j = joint.sum(dim=0)
    positive_indices = (joint > 0.0).nonzero(as_tuple=True)
    joint_positive = joint[positive_indices]
    marginal_i_positive = marginal_i[positive_indices[0]]
    marginal_j_positive = marginal_j[positive_indices[1]]
    assert torch.all(marginal_i_positive > 0.0)
    assert torch.all(marginal_j_positive > 0.0)
    mutual_information = torch.sum(
        joint_positive
        * (
            torch.log(joint_positive)
            - torch.log(marginal_i_positive)
            - torch.log(marginal_j_positive)
        )
    )
    return {
        "joint": joint,
        "marginal_i": marginal_i,
        "marginal_j": marginal_j,
        "mutual_information": mutual_information,
        "loss": -mutual_information,
    }


def plain(value: torch.Tensor):
    return value.detach().cpu().tolist()


def running_state(head: IICHead):
    return {
        "running_mean": plain(head.bn.running_mean),
        "running_var": plain(head.bn.running_var),
        "running_mean_active4": plain(head.bn.running_mean[:ACTIVE]),
        "running_var_active4": plain(head.bn.running_var[:ACTIVE]),
        "num_batches_tracked": int(head.bn.num_batches_tracked.item()),
    }


def parameter_gradients(head: IICHead):
    gradients = {
        "hidden_weight": head.hidden.weight.grad,
        "bn_gamma": head.bn.weight.grad,
        "bn_beta": head.bn.bias.grad,
        "output_weight": head.output.weight.grad,
        "output_bias": head.output.bias.grad,
    }
    assert all(value is not None for value in gradients.values())
    return {
        "full": {name: plain(value) for name, value in gradients.items()},
        "active4": {
            "hidden_weight": plain(gradients["hidden_weight"][:ACTIVE]),
            "bn_gamma": plain(gradients["bn_gamma"][:ACTIVE]),
            "bn_beta": plain(gradients["bn_beta"][:ACTIVE]),
            "output_weight": plain(gradients["output_weight"][:, :ACTIVE]),
            "output_bias": plain(gradients["output_bias"]),
        },
        "tail_max_abs": {
            "hidden_weight": float(gradients["hidden_weight"][ACTIVE:].abs().max()),
            "bn_gamma": float(gradients["bn_gamma"][ACTIVE:].abs().max()),
            "bn_beta": float(gradients["bn_beta"][ACTIVE:].abs().max()),
            "output_weight": float(gradients["output_weight"][:, ACTIVE:].abs().max()),
        },
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

    input_a = torch.tensor([
        [0.5, -1.0, 0.25],
        [-0.75, 0.5, 1.0],
        [1.25, 0.25, -0.5],
        [-0.25, -0.75, 0.75],
    ])
    input_b = -0.375 * input_a + torch.tensor([
        ((index % 5) - 2) * 0.125 for index in range(BATCH * INPUT_DIM)
    ]).reshape(BATCH, INPUT_DIM)

    pair = IICClassifierPair().cpu()
    initialize_pair(pair)
    pair.train()
    value_a = input_a.detach().clone().requires_grad_(True)
    value_b = input_b.detach().clone().requires_grad_(True)

    main_parameter_ptrs = {p.data_ptr() for p in pair.main_head.parameters()}
    over_parameter_ptrs = {p.data_ptr() for p in pair.overcluster_head.parameters()}
    main_buffer_ptrs = {b.data_ptr() for b in pair.main_head.buffers()}
    over_buffer_ptrs = {b.data_ptr() for b in pair.overcluster_head.buffers()}
    assert main_parameter_ptrs.isdisjoint(over_parameter_ptrs)
    assert main_buffer_ptrs.isdisjoint(over_buffer_ptrs)

    outputs_a = pair(value_a)  # main A, over A
    state_after_a = {
        "main": running_state(pair.main_head),
        "overcluster": running_state(pair.overcluster_head),
    }
    outputs_b = pair(value_b)  # main B, over B
    state_after_b = {
        "main": running_state(pair.main_head),
        "overcluster": running_state(pair.overcluster_head),
    }
    main_result = negative_mutual_information(
        outputs_a["main"], outputs_b["main"]
    )
    over_result = negative_mutual_information(
        outputs_a["overcluster"], outputs_b["overcluster"]
    )
    total_loss = main_result["loss"] + over_result["loss"]
    assert torch.isfinite(total_loss).item()
    total_loss.backward()
    assert state_after_a["main"]["num_batches_tracked"] == 1
    assert state_after_a["overcluster"]["num_batches_tracked"] == 1
    assert state_after_b["main"]["num_batches_tracked"] == 2
    assert state_after_b["overcluster"]["num_batches_tracked"] == 2

    payload = {
        "torch_version": str(torch.__version__),
        "dtype": "float32",
        "fixture": "MINI-C5B3 explicit pair, 4x3 input, 512 hidden, C=3, W=4",
        "forward_order": ["main_a", "over_a", "main_b", "over_b"],
        "input_a": plain(input_a),
        "input_b": plain(input_b),
        "probabilities": {
            "main_a": plain(outputs_a["main"]),
            "main_b": plain(outputs_b["main"]),
            "over_a": plain(outputs_a["overcluster"]),
            "over_b": plain(outputs_b["overcluster"]),
        },
        "main": {
            "joint": plain(main_result["joint"]),
            "marginal_i": plain(main_result["marginal_i"]),
            "marginal_j": plain(main_result["marginal_j"]),
            "mutual_information": float(main_result["mutual_information"].detach()),
            "loss": float(main_result["loss"].detach()),
        },
        "overcluster": {
            "joint": plain(over_result["joint"]),
            "marginal_i": plain(over_result["marginal_i"]),
            "marginal_j": plain(over_result["marginal_j"]),
            "mutual_information": float(over_result["mutual_information"].detach()),
            "loss": float(over_result["loss"].detach()),
        },
        "total_loss": float(total_loss.detach()),
        "input_gradient_a": plain(value_a.grad),
        "input_gradient_b": plain(value_b.grad),
        "parameter_gradients": {
            "main": parameter_gradients(pair.main_head),
            "overcluster": parameter_gradients(pair.overcluster_head),
        },
        "state_after_a": state_after_a,
        "state_after_b": state_after_b,
        "branch_parameter_independence_gate": True,
        "branch_buffer_independence_gate": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
