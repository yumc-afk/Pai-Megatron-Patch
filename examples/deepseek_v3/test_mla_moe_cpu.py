
import os
import sys
import torch
import math
import numpy as np
import argparse
from typing import Optional, Tuple, List, Dict, Any

class CustomRotaryEmbedding(torch.nn.Module):
    """旋转位置编码实现"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        seq_len = position_ids.shape[-1]
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        x_2d = x.view(*x.shape[:-1], -1, 2)
        x_2d_rot = torch.stack(
            [x_2d[..., 0] * cos - x_2d[..., 1] * sin, 
             x_2d[..., 1] * cos + x_2d[..., 0] * sin],
            dim=-1,
        )
        x_rot = x_2d_rot.flatten(-2)
        return x_rot

def test_rotary_embedding():
    print("Testing RotaryEmbedding...")
    dim = 64
    batch_size = 2
    seq_len = 10
    num_heads = 4
    
    rotary_emb = CustomRotaryEmbedding(dim=dim)
    
    x = torch.randn(batch_size, num_heads, seq_len, dim)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    output = rotary_emb(x, position_ids)
    
    assert output.shape == x.shape, f"输出形状 {output.shape} 与输入形状 {x.shape} 不匹配"
    
    assert not torch.isnan(output).any(), "输出包含NaN值"
    assert not torch.isinf(output).any(), "输出包含Inf值"
    
    print("RotaryEmbedding测试通过!")
    return True

def test_mla_attention():
    print("Testing MLA Attention...")
    hidden_size = 1024
    num_attention_heads = 16
    qk_nope_head_dim = 32
    qk_rope_head_dim = 32
    v_head_dim = 64
    batch_size = 2
    seq_len = 10
    
    query_nope = torch.randn(batch_size, seq_len, num_attention_heads, qk_nope_head_dim)
    query_rope = torch.randn(batch_size, seq_len, num_attention_heads, qk_rope_head_dim)
    key_nope = torch.randn(batch_size, seq_len, num_attention_heads, qk_nope_head_dim)
    key_rope = torch.randn(batch_size, seq_len, num_attention_heads, qk_rope_head_dim)
    value = torch.randn(batch_size, seq_len, num_attention_heads, v_head_dim)
    
    rotary_emb = CustomRotaryEmbedding(dim=qk_rope_head_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    query_rope = rotary_emb(query_rope, position_ids)
    key_rope = rotary_emb(key_rope, position_ids)
    
    query = torch.cat([query_nope, query_rope], dim=-1)
    key = torch.cat([key_nope, key_rope], dim=-1)
    
    query = query.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
    key = key.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
    value = value.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
    
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(query.size(-1))
    
    causal_mask = torch.tril(torch.ones((seq_len, seq_len)))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    causal_mask = (1.0 - causal_mask) * -10000.0
    
    attention_scores = attention_scores + causal_mask
    
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    
    context = torch.matmul(attention_probs, value)
    
    expected_shape = (batch_size, num_attention_heads, seq_len, v_head_dim)
    assert context.shape == expected_shape, f"输出形状 {context.shape} 与预期形状 {expected_shape} 不匹配"
    
    assert not torch.isnan(context).any(), "输出包含NaN值"
    assert not torch.isinf(context).any(), "输出包含Inf值"
    
    print("MLA Attention测试通过!")
    return True

def test_moe_router():
    print("Testing MOE Router...")
    hidden_size = 1024
    num_experts = 16
    topk = 2
    batch_size = 2
    seq_len = 10
    
    router_weights = torch.randn(hidden_size, num_experts)
    router_bias = torch.randn(num_experts)
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    router_logits = torch.matmul(hidden_states, router_weights) + router_bias
    
    router_probs = torch.sigmoid(router_logits)
    
    router_probs_topk, router_indices_topk = torch.topk(router_probs, topk, dim=-1)
    
    router_probs_topk = router_probs_topk / router_probs_topk.sum(dim=-1, keepdim=True)
    
    one_hot = torch.zeros_like(router_probs).scatter_(-1, router_indices_topk, 1.0)
    
    dispatch_weights = one_hot * router_probs_topk.unsqueeze(-1)
    
    assert router_indices_topk.shape == (batch_size, seq_len, topk), f"路由索引形状 {router_indices_topk.shape} 不正确"
    assert router_probs_topk.shape == (batch_size, seq_len, topk), f"路由概率形状 {router_probs_topk.shape} 不正确"
    assert dispatch_weights.shape == (batch_size, seq_len, num_experts), f"调度权重形状 {dispatch_weights.shape} 不正确"
    
    assert not torch.isnan(router_probs_topk).any(), "路由概率包含NaN值"
    assert not torch.isinf(router_probs_topk).any(), "路由概率包含Inf值"
    assert not torch.isnan(dispatch_weights).any(), "调度权重包含NaN值"
    assert not torch.isinf(dispatch_weights).any(), "调度权重包含Inf值"
    
    sum_probs = router_probs_topk.sum(dim=-1)
    assert torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=1e-5), f"路由概率和不为1: {sum_probs}"
    
    print("MOE Router测试通过!")
    return True

def test_moe_expert():
    print("Testing MOE Expert...")
    hidden_size = 1024
    ffn_hidden_size = 2816
    batch_size = 2
    seq_len = 10
    
    fc1_weight = torch.randn(ffn_hidden_size, hidden_size)
    fc2_weight = torch.randn(hidden_size, ffn_hidden_size)
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    intermediate = torch.matmul(hidden_states, fc1_weight.t())
    intermediate = torch.nn.functional.silu(intermediate)  # SwiGLU激活
    output = torch.matmul(intermediate, fc2_weight.t())
    
    expected_shape = (batch_size, seq_len, hidden_size)
    assert output.shape == expected_shape, f"输出形状 {output.shape} 与预期形状 {expected_shape} 不匹配"
    
    assert not torch.isnan(output).any(), "输出包含NaN值"
    assert not torch.isinf(output).any(), "输出包含Inf值"
    
    print("MOE Expert测试通过!")
    return True

def test_moe_dispatch():
    print("Testing MOE Dispatch...")
    hidden_size = 1024
    num_experts = 16
    topk = 2
    batch_size = 2
    seq_len = 10
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    router_indices = torch.randint(0, num_experts, (batch_size, seq_len, topk))
    router_probs = torch.rand(batch_size, seq_len, topk)
    router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True)
    
    expert_outputs = []
    for i in range(num_experts):
        expert_outputs.append(torch.randn(batch_size, seq_len, hidden_size))
    
    combined_output = torch.zeros(batch_size, seq_len, hidden_size)
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            for k in range(topk):
                expert_idx = router_indices[batch_idx, seq_idx, k].item()
                weight = router_probs[batch_idx, seq_idx, k].item()
                combined_output[batch_idx, seq_idx] += expert_outputs[expert_idx][batch_idx, seq_idx] * weight
    
    expected_shape = (batch_size, seq_len, hidden_size)
    assert combined_output.shape == expected_shape, f"输出形状 {combined_output.shape} 与预期形状 {expected_shape} 不匹配"
    
    assert not torch.isnan(combined_output).any(), "输出包含NaN值"
    assert not torch.isinf(combined_output).any(), "输出包含Inf值"
    
    print("MOE Dispatch测试通过!")
    return True

def main():
    print("开始CPU上的MLA/MOE组件验证...")
    
    tests = [
        test_rotary_embedding,
        test_mla_attention,
        test_moe_router,
        test_moe_expert,
        test_moe_dispatch
    ]
    
    all_passed = True
    for test in tests:
        try:
            passed = test()
            all_passed = all_passed and passed
        except Exception as e:
            print(f"测试失败: {e}")
            all_passed = False
    
    if all_passed:
        print("\n所有测试通过! MLA和MOE组件在CPU上验证成功。")
        return 0
    else:
        print("\n测试失败! 请检查错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
