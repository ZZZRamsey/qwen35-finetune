"""Exercise forward/backward without the optional causal-conv1d CUDA extension."""
import inspect
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import causal_conv1d_fn, torch_chunk_gated_delta_rule
from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet


def test_transformers_conv_fallback_forward_backward():
    torch.manual_seed(7)
    config = Qwen3_5MoeTextConfig(hidden_size=32, num_hidden_layers=1,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4)
    model = CPAwareGatedDeltaNet(config, 0)
    model.causal_conv1d_fn = inspect.unwrap(causal_conv1d_fn)
    model.chunk_gated_delta_rule = inspect.unwrap(torch_chunk_gated_delta_rule)
    with torch.no_grad():
        model.A_log.zero_()
        model.dt_bias.zero_()
    x = torch.randn(1, 8, 32, requires_grad=True)
    y = model(x, attention_mask=torch.ones(1, 8, dtype=torch.long))
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
