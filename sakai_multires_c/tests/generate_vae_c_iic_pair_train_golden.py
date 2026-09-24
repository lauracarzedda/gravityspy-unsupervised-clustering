"""Emit a two-step CPU PyTorch golden for one explicit MINI-C5B4 IIC pair.

This is standalone: no repository imports, dataset, labels, encoder, official
initializer, K=5 ensemble, or files written.  The fixture exactly matches
MINI-C5B3's pair golden, including its 512-wide BatchNorm and inactive tail.
One Adam owns both heads for two consecutive training steps.  The JSON on
stdout contains full tensors, not just active-channel summaries.
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
        # Dictionary values evaluate left-to-right: main, then overcluster.
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

        # Identical to generate_vae_c_iic_pair_golden.py / MINI-C5B3.
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
    """Exact Notebook06/C5B1 objective; no epsilon or symmetrization."""
    joint = probabilities_a.transpose(0, 1) @ probabilities_b / BATCH
    marginal_i = joint.sum(dim=1)
    marginal_j = joint.sum(dim=0)
    positive_indices = (joint > 0.0).nonzero(as_tuple=True)
    positive_joint = joint[positive_indices]
    positive_i = marginal_i[positive_indices[0]]
    positive_j = marginal_j[positive_indices[1]]
    assert torch.all(positive_i > 0.0)
    assert torch.all(positive_j > 0.0)
    mutual_information = torch.sum(
        positive_joint
        * (torch.log(positive_joint) - torch.log(positive_i) - torch.log(positive_j))
    )
    return mutual_information, -mutual_information


def plain(value: torch.Tensor):
    return value.detach().cpu().tolist()


def named_parameters(pair: IICClassifierPair):
    return {
        "main": {
            "hidden_weight": pair.main_head.hidden.weight,
            "bn_gamma": pair.main_head.bn.weight,
            "bn_beta": pair.main_head.bn.bias,
            "output_weight": pair.main_head.output.weight,
            "output_bias": pair.main_head.output.bias,
        },
        "overcluster": {
            "hidden_weight": pair.overcluster_head.hidden.weight,
            "bn_gamma": pair.overcluster_head.bn.weight,
            "bn_beta": pair.overcluster_head.bn.bias,
            "output_weight": pair.overcluster_head.output.weight,
            "output_bias": pair.overcluster_head.output.bias,
        },
    }


def tensor_tree(parameters, *, gradients: bool):
    return {
        branch: {
            name: plain(parameter.grad if gradients else parameter)
            for name, parameter in tensors.items()
        }
        for branch, tensors in parameters.items()
    }


def batchnorm_state(pair: IICClassifierPair):
    return {
        branch: {
            "running_mean": plain(head.bn.running_mean),
            "running_var": plain(head.bn.running_var),
            "num_batches_tracked": int(head.bn.num_batches_tracked.item()),
        }
        for branch, head in (
            ("main", pair.main_head),
            ("overcluster", pair.overcluster_head),
        )
    }


def adam_state(optimizer, parameters):
    return {
        branch: {
            name: {
                "m": plain(optimizer.state[parameter]["exp_avg"]),
                "v": plain(optimizer.state[parameter]["exp_avg_sq"]),
                "step": int(optimizer.state[parameter]["step"].item()),
            }
            for name, parameter in tensors.items()
        }
        for branch, tensors in parameters.items()
    }


def run_step(pair, optimizer, parameters, input_a, input_b, step_number):
    original_a = input_a.clone()
    original_b = input_b.clone()
    pair.train()
    optimizer.zero_grad(set_to_none=True)
    assert all(
        parameter.grad is None
        for branch in parameters.values()
        for parameter in branch.values()
    )

    outputs_a = pair(input_a)  # main A, over A
    bn_after_a = batchnorm_state(pair)
    outputs_b = pair(input_b)  # main B, over B
    bn_before_optimizer = batchnorm_state(pair)

    main_mi, main_loss = negative_mutual_information(
        outputs_a["main"], outputs_b["main"]
    )
    over_mi, over_loss = negative_mutual_information(
        outputs_a["overcluster"], outputs_b["overcluster"]
    )
    total_loss = main_loss + over_loss
    assert all(torch.isfinite(value).item() for value in (
        main_mi, main_loss, over_mi, over_loss, total_loss,
    ))
    total_loss.backward()

    parameter_list = [
        parameter
        for branch in parameters.values()
        for parameter in branch.values()
    ]
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        for parameter in parameter_list
    )
    gradients_before_step = tensor_tree(parameters, gradients=True)
    parameters_before_step = tensor_tree(parameters, gradients=False)

    optimizer.step()
    bn_after_optimizer = batchnorm_state(pair)
    assert bn_after_optimizer == bn_before_optimizer
    parameter_changed = any(
        not torch.equal(parameter, torch.tensor(
            parameters_before_step[branch][name], dtype=torch.float32
        ))
        for branch, tensors in parameters.items()
        for name, parameter in tensors.items()
    )
    assert parameter_changed
    assert len(optimizer.state) == 10
    assert all(int(optimizer.state[p]["step"].item()) == step_number
               for p in parameter_list)
    moment_updated = any(
        torch.count_nonzero(optimizer.state[p]["exp_avg"]).item() > 0
        and torch.count_nonzero(optimizer.state[p]["exp_avg_sq"]).item() > 0
        for p in parameter_list
    )
    assert moment_updated
    assert torch.equal(input_a, original_a)
    assert torch.equal(input_b, original_b)

    return {
        "step": step_number,
        "zero_grad_set_to_none_gate": True,
        "input_unchanged_gate": True,
        "parameter_changed_gate": parameter_changed,
        "adam_m_v_updated_gate": moment_updated,
        "main": {
            "mutual_information": float(main_mi.detach()),
            "loss": float(main_loss.detach()),
        },
        "overcluster": {
            "mutual_information": float(over_mi.detach()),
            "loss": float(over_loss.detach()),
        },
        "total_loss": float(total_loss.detach()),
        "gradients_before_step": gradients_before_step,
        "parameters_after_step": tensor_tree(parameters, gradients=False),
        "adam_state_after_step": adam_state(optimizer, parameters),
        "optimizer_step": step_number,
        "bn_after_view_a": bn_after_a,
        "bn_before_optimizer": bn_before_optimizer,
        "bn_after_optimizer": bn_after_optimizer,
        "bn_unchanged_by_optimizer_gate": True,
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
    original_a = input_a.clone()
    original_b = input_b.clone()

    pair = IICClassifierPair().cpu()
    initialize_pair(pair)
    parameters = named_parameters(pair)
    main_ptrs = {p.data_ptr() for p in parameters["main"].values()}
    over_ptrs = {p.data_ptr() for p in parameters["overcluster"].values()}
    main_buffer_ptrs = {b.data_ptr() for b in pair.main_head.buffers()}
    over_buffer_ptrs = {b.data_ptr() for b in pair.overcluster_head.buffers()}
    assert main_ptrs.isdisjoint(over_ptrs)
    assert main_buffer_ptrs.isdisjoint(over_buffer_ptrs)

    optimizer = torch.optim.Adam(
        pair.parameters(),
        lr=5.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    assert len(optimizer.state) == 0  # PyTorch lazily allocates zero m/v.
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(owned) == 10
    assert {p.data_ptr() for p in owned} == main_ptrs | over_ptrs
    initial_bn = batchnorm_state(pair)
    initial_parameters = tensor_tree(parameters, gradients=False)

    steps = [
        run_step(pair, optimizer, parameters, input_a, input_b, number)
        for number in (1, 2)
    ]
    assert torch.equal(input_a, original_a)
    assert torch.equal(input_b, original_b)
    assert steps[0]["bn_after_optimizer"]["main"]["num_batches_tracked"] == 2
    assert steps[0]["bn_after_optimizer"]["overcluster"]["num_batches_tracked"] == 2
    assert steps[1]["bn_after_optimizer"]["main"]["num_batches_tracked"] == 4
    assert steps[1]["bn_after_optimizer"]["overcluster"]["num_batches_tracked"] == 4

    payload = {
        "torch_version": str(torch.__version__),
        "dtype": "float32",
        "fixture": "MINI-C5B3 explicit pair, B=4, dz=3, hidden=512, C=3, W=4",
        "forward_order": ["main_a", "over_a", "main_b", "over_b"],
        "optimizer": {
            "type": "Adam",
            "lr": 5.0e-4,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "parameter_count": 10,
            "initial_state_entries": 0,
            "initial_m_v_semantically_zero": True,
        },
        "input_a": plain(input_a),
        "input_b": plain(input_b),
        "input_unchanged_gate": True,
        "branch_parameter_independence_gate": True,
        "branch_buffer_independence_gate": True,
        "initial_parameters": initial_parameters,
        "initial_bn": initial_bn,
        "steps": steps,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
