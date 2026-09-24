"""Generate the MINI-C5B2 deterministic PyTorch golden on CPU.

This is a reference-only helper.  It uses explicit values rather than the
official classifier initializer, reads no project data, and writes no files.
The productive CTest embeds the emitted values and does not require PyTorch.
"""

import json

import torch
import torch.nn as nn


HIDDEN = 512


class IICHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dim, HIDDEN, bias=False)
        self.bn = nn.BatchNorm1d(
            HIDDEN,
            eps=1.0e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.relu = nn.ReLU(inplace=False)
        self.output = nn.Linear(HIDDEN, output_dim, bias=True)

    def stages(self, value: torch.Tensor):
        linear1 = self.hidden(value)
        batch_norm = self.bn(linear1)
        relu = self.relu(batch_norm)
        logits = self.output(relu)
        probabilities = torch.softmax(logits, dim=1)
        return linear1, batch_norm, relu, logits, probabilities


def active4(value: torch.Tensor):
    if value.ndim == 1:
        return value[:4].detach().cpu().tolist()
    return value[:, :4].detach().cpu().tolist()


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

    head = IICHead(3, 3).cpu()
    with torch.no_grad():
        head.hidden.weight.zero_()
        head.hidden.weight[:4].copy_(torch.tensor([
            [0.5, -0.25, 0.125],
            [-0.375, 0.25, 0.5],
            [0.25, 0.75, -0.5],
            [-0.125, 0.5, 0.375],
        ]))
        head.bn.weight.fill_(1.0)
        head.bn.weight[:4].copy_(torch.tensor([1.125, -0.875, 0.75, 1.25]))
        head.bn.bias.fill_(-1.0)
        head.bn.bias[:4].copy_(torch.tensor([0.125, 0.25, -0.0625, -0.125]))
        head.bn.running_mean.zero_()
        head.bn.running_mean[:4].copy_(torch.tensor([0.25, -0.125, 0.375, -0.25]))
        head.bn.running_var.fill_(1.0)
        head.bn.running_var[:4].copy_(torch.tensor([1.5, 0.75, 2.0, 1.25]))
        head.bn.num_batches_tracked.zero_()
        head.output.weight.zero_()
        head.output.weight[:, :4].copy_(torch.tensor([
            [0.5, -0.25, 0.125, 0.0625],
            [-0.125, 0.375, 0.25, -0.5],
            [0.25, 0.125, -0.375, 0.625],
        ]))
        head.output.bias.copy_(torch.tensor([0.0625, -0.125, 0.1875]))

    value = torch.tensor([
        [0.5, -1.0, 0.25],
        [-0.75, 0.5, 1.0],
        [1.25, 0.25, -0.5],
        [-0.25, -0.75, 0.75],
    ], requires_grad=True)
    coefficient = torch.tensor([
        [0.25, -0.375, 0.5],
        [-0.5, 0.625, 0.125],
        [0.375, 0.25, -0.625],
        [0.75, -0.25, 0.3125],
    ])

    head.train()
    stages = head.stages(value)
    state_after_train1 = (
        head.bn.running_mean.clone(),
        head.bn.running_var.clone(),
        int(head.bn.num_batches_tracked.item()),
    )
    loss = (stages[-1] * coefficient).sum()
    loss.backward()

    with torch.no_grad():
        head.stages(value.detach())
        state_after_train2 = (
            head.bn.running_mean.clone(),
            head.bn.running_var.clone(),
            int(head.bn.num_batches_tracked.item()),
        )
        head.eval()
        eval_stages = head.stages(value.detach())

    payload = {
        "torch_version": torch.__version__,
        "dtype": "float32",
        "linear1_active4": active4(stages[0]),
        "batch_norm_active4": active4(stages[1]),
        "relu_active4": active4(stages[2]),
        "logits": stages[3].detach().cpu().tolist(),
        "probabilities": stages[4].detach().cpu().tolist(),
        "loss": float(loss.detach().item()),
        "running_mean_after_train1_active4": active4(state_after_train1[0]),
        "running_var_after_train1_active4": active4(state_after_train1[1]),
        "num_batches_after_train1": state_after_train1[2],
        "running_mean_after_train2_active4": active4(state_after_train2[0]),
        "running_var_after_train2_active4": active4(state_after_train2[1]),
        "num_batches_after_train2": state_after_train2[2],
        "input_grad": value.grad.detach().cpu().tolist(),
        "linear1_weight_grad_active4": head.hidden.weight.grad[:4].detach().cpu().tolist(),
        "bn_gamma_grad_active4": active4(head.bn.weight.grad),
        "bn_beta_grad_active4": active4(head.bn.bias.grad),
        "linear2_weight_grad_active4": head.output.weight.grad[:, :4].detach().cpu().tolist(),
        "linear2_bias_grad": head.output.bias.grad.detach().cpu().tolist(),
        "eval_batch_norm_active4": active4(eval_stages[1]),
        "eval_relu_active4": active4(eval_stages[2]),
        "eval_logits": eval_stages[3].detach().cpu().tolist(),
        "eval_probabilities": eval_stages[4].detach().cpu().tolist(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
