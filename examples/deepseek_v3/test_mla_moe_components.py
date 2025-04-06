
import os
import sys
import torch
import math
import numpy as np
import argparse
from typing import Optional, Tuple, List, Dict, Any

class CustomRotaryEmbedding(torch.nn.Module):
    """简化版旋转位置编码，基于DeepSeek V3的实现"""
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        self._update_cos_sin_cache(max_position_embeddings)
    
    def _update_cos_sin_cache(self, seq_len: int):
        """更新余弦和正弦缓存"""
        self.max_seq_len_cached = seq_len
        
        t = torch.arange(seq_len, dtype=torch.float32)
        t = t / self.scaling_factor
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]
    
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """前向传播"""
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_seq_len_cached:
            self._update_cos_sin_cache(seq_len)
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device)
        
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        cos = cos.expand(batch_size, 1, seq_len, self.dim)
        sin = sin.expand(batch_size, 1, seq_len, self.dim)
        
        return cos, sin

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """应用旋转位置编码"""
    q_embed_dim = q.shape[-1]
    k_embed_dim = k.shape[-1]
    
    q_half1, q_half2 = torch.split(q, q_embed_dim // 2, dim=-1)
    k_half1, k_half2 = torch.split(k, k_embed_dim // 2, dim=-1)
    
    if cos.dim() == 4 and q.dim() == 4:
        cos = cos.squeeze(0).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(0).squeeze(0)  # [seq_len, dim]
        
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        
        cos = cos.expand(q.size(0), q.size(2), -1, -1)  # [batch, heads, seq_len, dim]
        sin = sin.expand(q.size(0), q.size(2), -1, -1)  # [batch, heads, seq_len, dim]
        
        cos = cos.transpose(1, 2)  # [batch, seq_len, heads, dim]
        sin = sin.transpose(1, 2)  # [batch, seq_len, heads, dim]
    
    q_rotated = torch.cat([
        q_half1 * cos - q_half2 * sin,
        q_half2 * cos + q_half1 * sin
    ], dim=-1)
    
    k_rotated = torch.cat([
        k_half1 * cos - k_half2 * sin,
        k_half2 * cos + k_half1 * sin
    ], dim=-1)
    
    return q_rotated, k_rotated

class CustomMultiLatentAttention(torch.nn.Module):
    """简化版多潜在注意力机制，基于DeepSeek V3的实现"""
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        qk_head_dim: int = 64,
        qk_pos_emb_head_dim: int = 64,
        v_head_dim: int = 128,
        max_position_embeddings: int = 2048,
        rotary_base: int = 10000,
        rotary_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.qk_head_dim = qk_head_dim
        self.qk_pos_emb_head_dim = qk_pos_emb_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_head_dim + qk_pos_emb_head_dim
        
        self.query_projection_size = v_head_dim * num_attention_heads
        
        self.softmax_scale = 1.0 / math.sqrt(self.q_head_dim)
        
        self.rotary_pos_emb = CustomRotaryEmbedding(
            dim=qk_pos_emb_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rotary_base,
            scaling_factor=rotary_scaling_factor,
        )
        
        self.q_proj = torch.nn.Linear(hidden_size, num_attention_heads * self.q_head_dim, bias=False)
        self.kv_down_proj = torch.nn.Linear(hidden_size, 512 + qk_pos_emb_head_dim, bias=False)
        self.kv_layernorm = torch.nn.LayerNorm(512)
        self.kv_up_proj = torch.nn.Linear(512, num_attention_heads * (qk_head_dim + v_head_dim), bias=False)
        self.output_proj = torch.nn.Linear(self.query_projection_size, hidden_size, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """前向传播"""
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.q_head_dim)
        
        q_no_pe, q_pos_emb = torch.split(
            q, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=-1
        )
        
        kv_combined = self.kv_down_proj(hidden_states)
        kv_compressed, k_pos_emb = torch.split(
            kv_combined, [512, self.qk_pos_emb_head_dim], dim=-1
        )
        
        kv = self.kv_up_proj(self.kv_layernorm(kv_compressed))
        kv = kv.view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.qk_head_dim + self.v_head_dim,
        )
        
        k_no_pe, value = torch.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1)
        
        cos, sin = self.rotary_pos_emb(q_pos_emb)
        
        k_pos_emb = k_pos_emb.unsqueeze(2).expand(-1, -1, self.num_attention_heads, -1)
        
        q_pos_emb_rotated, k_pos_emb_rotated = apply_rotary_pos_emb(q_pos_emb, k_pos_emb, cos, sin)
        
        query = torch.cat([q_no_pe, q_pos_emb_rotated], dim=-1)
        key = torch.cat([k_no_pe, k_pos_emb_rotated], dim=-1)
        
        query = query.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        key = key.permute(0, 2, 1, 3)      # [batch_size, num_heads, seq_len, head_dim]
        value = value.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * self.softmax_scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        context = torch.matmul(attention_probs, value)
        
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, -1)
        
        output = self.output_proj(context)
        
        return output

class CustomMoERouter(torch.nn.Module):
    """简化版MOE路由器，基于DeepSeek V3的实现"""
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_experts: int = 16,
        router_topk: int = 2,
        router_score_function: str = "sigmoid",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router_topk = router_topk
        self.router_score_function = router_score_function
        
        self.router = torch.nn.Linear(hidden_size, num_experts, bias=True)
    
    def forward(self, hidden_states: torch.Tensor):
        """前向传播"""
        batch_size, seq_len, _ = hidden_states.shape
        
        router_logits = self.router(hidden_states)
        
        if self.router_score_function == "sigmoid":
            router_probs = torch.sigmoid(router_logits)
        elif self.router_score_function == "softmax":
            router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
        else:
            raise ValueError(f"不支持的路由器评分函数: {self.router_score_function}")
        
        router_probs_topk, router_indices_topk = torch.topk(
            router_probs, self.router_topk, dim=-1
        )
        
        router_probs_topk = router_probs_topk / router_probs_topk.sum(dim=-1, keepdim=True)
        
        routing_map = torch.zeros_like(router_probs)
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(self.router_topk):
                    expert_idx = router_indices_topk[b, s, k]
                    weight = router_probs_topk[b, s, k]
                    routing_map[b, s, expert_idx] = weight
        
        return router_probs_topk, router_indices_topk, routing_map

class CustomMoEExpert(torch.nn.Module):
    """简化版MOE专家，基于DeepSeek V3的实现"""
    
    def __init__(
        self,
        hidden_size: int = 1024,
        ffn_hidden_size: int = 2816,
        activation_func: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.activation_func = activation_func
        
        self.fc1 = torch.nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.fc2 = torch.nn.Linear(ffn_hidden_size, hidden_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor):
        """前向传播"""
        intermediate = self.fc1(hidden_states)
        
        if self.activation_func == "silu":
            intermediate = torch.nn.functional.silu(intermediate)
        elif self.activation_func == "gelu":
            intermediate = torch.nn.functional.gelu(intermediate)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation_func}")
        
        output = self.fc2(intermediate)
        
        return output

class CustomMoE(torch.nn.Module):
    """简化版MOE层，基于DeepSeek V3的实现"""
    
    def __init__(
        self,
        hidden_size: int = 1024,
        ffn_hidden_size: int = 2816,
        num_experts: int = 16,
        router_topk: int = 2,
        router_score_function: str = "sigmoid",
        activation_func: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_experts = num_experts
        self.router_topk = router_topk
        
        self.router = CustomMoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            router_topk=router_topk,
            router_score_function=router_score_function,
        )
        
        self.experts = torch.nn.ModuleList([
            CustomMoEExpert(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                activation_func=activation_func,
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, hidden_states: torch.Tensor):
        """前向传播"""
        batch_size, seq_len, _ = hidden_states.shape
        
        router_probs_topk, router_indices_topk, routing_map = self.router(hidden_states)
        
        output = torch.zeros_like(hidden_states)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (routing_map[:, :, expert_idx] > 0).float().unsqueeze(-1)
            
            if expert_mask.sum() == 0:
                continue
            
            expert_output = self.experts[expert_idx](hidden_states)
            
            weighted_output = expert_output * routing_map[:, :, expert_idx].unsqueeze(-1)
            
            output += weighted_output
        
        return output, router_probs_topk, router_indices_topk

class CustomTransformerLayer(torch.nn.Module):
    """简化版Transformer层，基于DeepSeek V3的实现"""
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        qk_head_dim: int = 64,
        qk_pos_emb_head_dim: int = 64,
        v_head_dim: int = 128,
        ffn_hidden_size: int = 2816,
        num_experts: int = 16,
        router_topk: int = 2,
        use_moe: bool = True,
        max_position_embeddings: int = 2048,
        rotary_base: int = 10000,
        rotary_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_moe = use_moe
        
        self.input_layernorm = torch.nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_size)
        
        self.attention = CustomMultiLatentAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            qk_head_dim=qk_head_dim,
            qk_pos_emb_head_dim=qk_pos_emb_head_dim,
            v_head_dim=v_head_dim,
            max_position_embeddings=max_position_embeddings,
            rotary_base=rotary_base,
            rotary_scaling_factor=rotary_scaling_factor,
        )
        
        if use_moe:
            self.mlp = CustomMoE(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                num_experts=num_experts,
                router_topk=router_topk,
            )
        else:
            self.mlp = CustomMoEExpert(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """前向传播"""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attention_output
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.use_moe:
            mlp_output, router_probs, router_indices = self.mlp(hidden_states)
            hidden_states = residual + mlp_output
            return hidden_states, router_probs, router_indices
        else:
            mlp_output = self.mlp(hidden_states)
            hidden_states = residual + mlp_output
            return hidden_states
        
def test_mla_component():
    """测试MLA组件"""
    print("\n===== 测试MLA组件 =====")
    
    batch_size = 2
    seq_len = 16
    hidden_size = 128
    num_heads = 4
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    attention_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    attention_mask = (1.0 - attention_mask) * -10000.0
    
    mla = CustomMultiLatentAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        qk_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        max_position_embeddings=64,
    )
    
    try:
        output = mla(hidden_states, attention_mask)
        
        expected_shape = (batch_size, seq_len, hidden_size)
        assert output.shape == expected_shape, f"输出形状 {output.shape} 与预期形状 {expected_shape} 不匹配"
        
        assert not torch.isnan(output).any(), "输出包含NaN值"
        assert not torch.isinf(output).any(), "输出包含Inf值"
        
        print("MLA组件测试通过!")
        return True
    except Exception as e:
        print(f"MLA组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_moe_component():
    """测试MOE组件"""
    print("\n===== 测试MOE组件 =====")
    
    batch_size = 2
    seq_len = 16
    hidden_size = 128
    num_experts = 4
    router_topk = 2
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    moe = CustomMoE(
        hidden_size=hidden_size,
        ffn_hidden_size=256,
        num_experts=num_experts,
        router_topk=router_topk,
    )
    
    try:
        output, router_probs, router_indices = moe(hidden_states)
        
        expected_shape = (batch_size, seq_len, hidden_size)
        assert output.shape == expected_shape, f"输出形状 {output.shape} 与预期形状 {expected_shape} 不匹配"
        
        expected_router_probs_shape = (batch_size, seq_len, router_topk)
        assert router_probs.shape == expected_router_probs_shape, f"路由器概率形状 {router_probs.shape} 与预期形状 {expected_router_probs_shape} 不匹配"
        
        expected_router_indices_shape = (batch_size, seq_len, router_topk)
        assert router_indices.shape == expected_router_indices_shape, f"路由器索引形状 {router_indices.shape} 与预期形状 {expected_router_indices_shape} 不匹配"
        
        sum_probs = router_probs.sum(dim=-1)
        assert torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=1e-5), f"路由器概率和不为1: {sum_probs}"
        
        assert not torch.isnan(output).any(), "输出包含NaN值"
        assert not torch.isinf(output).any(), "输出包含Inf值"
        
        print("MOE组件测试通过!")
        return True
    except Exception as e:
        print(f"MOE组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transformer_layer():
    """测试Transformer层"""
    print("\n===== 测试Transformer层 =====")
    
    batch_size = 2
    seq_len = 16
    hidden_size = 128
    num_heads = 4
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    attention_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    attention_mask = (1.0 - attention_mask) * -10000.0
    
    transformer_moe = CustomTransformerLayer(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        qk_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        ffn_hidden_size=256,
        num_experts=4,
        router_topk=2,
        use_moe=True,
        max_position_embeddings=64,
    )
    
    transformer_no_moe = CustomTransformerLayer(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        qk_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        ffn_hidden_size=256,
        num_experts=4,
        router_topk=2,
        use_moe=False,
        max_position_embeddings=64,
    )
    
    try:
        output_moe, router_probs, router_indices = transformer_moe(hidden_states, attention_mask)
        
        expected_shape = (batch_size, seq_len, hidden_size)
        assert output_moe.shape == expected_shape, f"MOE输出形状 {output_moe.shape} 与预期形状 {expected_shape} 不匹配"
        
        assert not torch.isnan(output_moe).any(), "MOE输出包含NaN值"
        assert not torch.isinf(output_moe).any(), "MOE输出包含Inf值"
        
        print("带MOE的Transformer层测试通过!")
    except Exception as e:
        print(f"带MOE的Transformer层测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        output_no_moe = transformer_no_moe(hidden_states, attention_mask)
        
        expected_shape = (batch_size, seq_len, hidden_size)
        assert output_no_moe.shape == expected_shape, f"非MOE输出形状 {output_no_moe.shape} 与预期形状 {expected_shape} 不匹配"
        
        assert not torch.isnan(output_no_moe).any(), "非MOE输出包含NaN值"
        assert not torch.isinf(output_no_moe).any(), "非MOE输出包含Inf值"
        
        print("不带MOE的Transformer层测试通过!")
        return True
    except Exception as e:
        print(f"不带MOE的Transformer层测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始CPU上的MLA/MOE组件验证...")
    
    tests = [
        test_mla_component,
        test_moe_component,
        test_transformer_layer,
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
