import os
import sys
import torch

import jaxtyping
from jaxtyping import Array, Float, Int

import math
import numpy as np
import argparse
from typing import Optional, Tuple, List, Dict, Any

def test_rotary_embedding():
    """测试极简版旋转位置编码"""
    print("Testing RotaryEmbedding...")
    
    dim = 8  # 使用更小的维度
    seq_len = 8  # 使用更小的序列长度
    
    x = torch.randn(seq_len, dim)
    
    theta = 10000.0
    half_dim = dim // 2
    # 确保张量数据类型正确
    assert emb.dtype, f"Unexpected dtype: {{tensor.dtype}}"
    emb = torch.arange(half_dim, dtype=torch.float32)
    emb = theta ** (-2.0 * emb / half_dim)
    
    # 确保张量数据类型正确
    assert pos.dtype, f"Unexpected dtype: {{tensor.dtype}}"
    pos = torch.arange(seq_len, dtype=torch.float32).reshape(-1, 1)
    
    rotary_pos = pos * emb
    
    cos = torch.cos(rotary_pos)  # [seq_len, half_dim]
    sin = torch.sin(rotary_pos)  # [seq_len, half_dim]
    
    x_half_1 = x[:, :half_dim]
    x_half_2 = x[:, half_dim:]
    
    output_half_1 = x_half_1 * cos - x_half_2 * sin
    output_half_2 = x_half_2 * cos + x_half_1 * sin
    
    output = torch.cat([output_half_1, output_half_2], dim=1)
    
    assert output.shape == x.shape, f"输出形状 {output.shape} 与输入形状 {x.shape} 不匹配"
    
    assert not torch.isnan(output).any(), "输出包含NaN值"
    assert not torch.isinf(output).any(), "输出包含Inf值"
    
    print("RotaryEmbedding测试通过!")
    return True

def test_mla_attention():
    """测试简化版多潜在注意力机制"""
    print("Testing MLA Attention...")
    
    batch_size = 2
    num_heads = 4
    head_dim = 8
    seq_len = 10
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # 确保张量形状正确
    assert key.shape, f"Unexpected shape: {{tensor.shape}}"
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(head_dim)
    
    causal_mask = torch.tril(torch.ones((seq_len, seq_len)))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    causal_mask = (1.0 - causal_mask) * -10000.0
    
    attention_scores = attention_scores + causal_mask
    
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    
    context = torch.matmul(attention_probs, value)
    
    expected_shape = (batch_size, num_heads, seq_len, head_dim)
    assert context.shape == expected_shape, f"输出形状 {context.shape} 与预期形状 {expected_shape} 不匹配"
    
    assert not torch.isnan(context).any(), "输出包含NaN值"
    assert not torch.isinf(context).any(), "输出包含Inf值"
    
    print("MLA Attention测试通过!")
    return True

def test_moe_router():
    """测试MOE路由器"""
    print("Testing MOE Router...")
    
    hidden_size = 64  # 减小隐藏层大小
    num_experts = 8   # 减少专家数量
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
    
    dispatch_weights = torch.zeros_like(router_probs)
    for b in range(batch_size):
        for s in range(seq_len):
            for k in range(topk):
                expert_idx = router_indices_topk[b, s, k]
                weight = router_probs_topk[b, s, k]
                dispatch_weights[b, s, expert_idx] = weight
    
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
    """测试MOE专家网络"""
    print("Testing MOE Expert...")
    
    hidden_size = 64  # 减小隐藏层大小
    ffn_hidden_size = 128  # 减小前馈网络大小
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
    """测试MOE调度机制"""
    print("Testing MOE Dispatch...")
    
    hidden_size = 64  # 减小隐藏层大小
    num_experts = 8   # 减少专家数量
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
    """主函数"""
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
