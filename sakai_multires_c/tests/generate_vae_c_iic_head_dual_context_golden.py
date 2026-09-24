"""Generate a deterministic dual-TRAIN-context IIC-head PyTorch golden.

Run this standalone on a CPU environment with PyTorch (for example Colab).
The script reads no project data, writes no files, and does not implement the
classifier pair or IIC loss. Both views use one head, with two forward calls
before a single backward of two simple scalar losses.
"""

import copy
import json

import torch
import torch.nn as nn


HIDDEN = 512


class IICHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(3, HIDDEN, bias=False)
        self.bn = nn.BatchNorm1d(
            HIDDEN,
            eps=1.0e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.relu = nn.ReLU(inplace=False)
        self.linear2 = nn.Linear(HIDDEN, 3, bias=True)

    def forward(self, value: torch.Tensor):
        linear1 = self.linear1(value)
        batch_norm = self.bn(linear1)
        relu = self.relu(batch_norm)
        logits = self.linear2(relu)
        probabilities = torch.softmax(logits, dim=1)
        return linear1, batch_norm, relu, logits, probabilities


def make_head() -> IICHead:
    head = IICHead().cpu()
    with torch.no_grad():
        head.linear1.weight.zero_()
        head.linear1.weight[:4].copy_(torch.tensor([
            [0.5, -0.25, 0.125],
            [-0.375, 0.25, 0.5],
            [0.25, 0.75, -0.5],
            [-0.125, 0.5, 0.375],
        ]))
        head.bn.weight.fill_(1.0)
        head.bn.weight[:4].copy_(torch.tensor([
            1.125, -0.875, 0.75, 1.25,
        ]))
        head.bn.bias.fill_(-1.0)
        head.bn.bias[:4].copy_(torch.tensor([
            0.125, 0.25, -0.0625, -0.125,
        ]))
        head.bn.running_mean.zero_()
        head.bn.running_mean[:4].copy_(torch.tensor([
            0.25, -0.125, 0.375, -0.25,
        ]))
        head.bn.running_var.fill_(1.0)
        head.bn.running_var[:4].copy_(torch.tensor([
            1.5, 0.75, 2.0, 1.25,
        ]))
        head.bn.num_batches_tracked.zero_()
        head.linear2.weight.zero_()
        head.linear2.weight[:, :4].copy_(torch.tensor([
            [0.5, -0.25, 0.125, 0.0625],
            [-0.125, 0.375, 0.25, -0.5],
            [0.25, 0.125, -0.375, 0.625],
        ]))
        head.linear2.bias.copy_(torch.tensor([
            0.0625, -0.125, 0.1875,
        ]))
    head.train()
    return head


def active4(value: torch.Tensor):
    if value.ndim == 1:
        return value[:4].detach().cpu().tolist()
    return value[:, :4].detach().cpu().tolist()


def tail_max_abs(value: torch.Tensor) -> float:
    tail = value[4:] if value.ndim == 1 else value[:, 4:]
    return float(tail.detach().abs().max().item()) if tail.numel() else 0.0


def running_state(head: IICHead):
    return {
        "running_mean_active4": active4(head.bn.running_mean),
        "running_var_active4": active4(head.bn.running_var),
        "running_mean_tail_max_abs": tail_max_abs(head.bn.running_mean),
        "running_var_tail_min": float(head.bn.running_var[4:].min().item()),
        "num_batches_tracked": int(head.bn.num_batches_tracked.item()),
    }


def stages_payload(stages):
    return {
        "linear1_active4": stages[0][:, :4].detach().cpu().tolist(),
        "batch_norm_active4": stages[1][:, :4].detach().cpu().tolist(),
        "relu_active4": stages[2][:, :4].detach().cpu().tolist(),
        "logits": stages[3].detach().cpu().tolist(),
        "probabilities": stages[4].detach().cpu().tolist(),
    }


def gradient_payload(head: IICHead):
    return {
        "linear1_weight_active4": head.linear1.weight.grad[:4].detach().cpu().tolist(),
        "linear1_weight_tail_max_abs": float(
            head.linear1.weight.grad[4:].detach().abs().max().item()
        ),
        "bn_gamma_active4": active4(head.bn.weight.grad),
        "bn_gamma_tail_max_abs": tail_max_abs(head.bn.weight.grad),
        "bn_beta_active4": active4(head.bn.bias.grad),
        "bn_beta_tail_max_abs": tail_max_abs(head.bn.bias.grad),
        "linear2_weight_active4": active4(head.linear2.weight.grad),
        "linear2_weight_tail_max_abs": tail_max_abs(head.linear2.weight.grad),
        "linear2_bias": head.linear2.bias.grad.detach().cpu().tolist(),
    }


def branch_reference(
    pristine_head: IICHead,
    input_data: torch.Tensor,
    coefficient: torch.Tensor,
):
    head = copy.deepcopy(pristine_head)
    value = input_data.detach().clone().requires_grad_(True)
    stages = head(value)
    loss = (stages[-1] * coefficient).sum()
    loss.backward()
    return head, value, stages, loss


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
    input_b = (
        -0.375 * input_a
        + torch.tensor(
            [((index % 5) - 2) * 0.125 for index in range(12)]
        ).reshape(4, 3)
    )
    coefficient_a = torch.tensor([
        [0.25, -0.375, 0.5],
        [-0.5, 0.625, 0.125],
        [0.375, 0.25, -0.625],
        [0.75, -0.25, 0.3125],
    ])
    coefficient_b = (
        -0.625 * coefficient_a
        + torch.tensor(
            [((index % 3) - 1) * 0.03125 for index in range(12)]
        ).reshape(4, 3)
    )

    head = make_head()
    pristine_head = copy.deepcopy(head)
    value_a = input_a.detach().clone().requires_grad_(True)
    value_b = input_b.detach().clone().requires_grad_(True)
    stages_a = head(value_a)
    state_after_a = running_state(head)
    stages_b = head(value_b)
    state_after_b = running_state(head)
    loss_a = (stages_a[-1] * coefficient_a).sum()
    loss_b = (stages_b[-1] * coefficient_b).sum()
    total_loss = loss_a + loss_b
    total_loss.backward()

    single_a, single_value_a, single_stages_a, single_loss_a = branch_reference(
        pristine_head, input_a, coefficient_a
    )
    single_b, single_value_b, single_stages_b, single_loss_b = branch_reference(
        pristine_head, input_b, coefficient_b
    )
    for pair_param, a_param, b_param in zip(
        head.parameters(), single_a.parameters(), single_b.parameters()
    ):
        torch.testing.assert_close(
            pair_param.grad,
            a_param.grad + b_param.grad,
            rtol=2.0e-5,
            atol=1.0e-6,
        )
    torch.testing.assert_close(value_a.grad, single_value_a.grad, rtol=2.0e-5, atol=1.0e-6)
    torch.testing.assert_close(value_b.grad, single_value_b.grad, rtol=2.0e-5, atol=1.0e-6)
    torch.testing.assert_close(stages_a[-1], single_stages_a[-1], rtol=2.0e-5, atol=1.0e-6)
    torch.testing.assert_close(stages_b[-1], single_stages_b[-1], rtol=2.0e-5, atol=1.0e-6)
    torch.testing.assert_close(loss_a, single_loss_a, rtol=2.0e-5, atol=1.0e-6)
    torch.testing.assert_close(loss_b, single_loss_b, rtol=2.0e-5, atol=1.0e-6)
    assert state_after_a["num_batches_tracked"] == 1
    assert state_after_b["num_batches_tracked"] == 2

    payload = {
        "torch_version": torch.__version__,
        "dtype": "float32",
        "input_a": input_a.tolist(),
        "input_b": input_b.tolist(),
        "coefficient_a": coefficient_a.tolist(),
        "coefficient_b": coefficient_b.tolist(),
        "view_a": stages_payload(stages_a),
        "view_b": stages_payload(stages_b),
        "state_after_a": state_after_a,
        "state_after_b": state_after_b,
        "loss_a": float(loss_a.detach().item()),
        "loss_b": float(loss_b.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "input_gradient_a": value_a.grad.detach().cpu().tolist(),
        "input_gradient_b": value_b.grad.detach().cpu().tolist(),
        "parameter_gradients_from_a": gradient_payload(single_a),
        "parameter_gradients_from_b": gradient_payload(single_b),
        "parameter_gradients_total": gradient_payload(head),
        "gradient_sum_gate": True,
        "input_gradient_independence_gate": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
