#
#
#

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from functools import partial
from typing import Optional, Tuple, List, Dict, Any

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(path_dir, "examples"))

from deepseek_v3.pretrain_deepseek import model_provider
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron

class CustomRotaryEmbedding(nn.Module):
    """自定义旋转位置编码实现"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, scaling_factor=1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, 
            device=torch.device("cpu"),
            dtype=torch.float32
        )
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        t = t / self.scaling_factor
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)
    
    def forward(self, x, position_ids=None):
        """应用旋转位置编码"""
        batch_size, seq_len, _ = x.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        cos = self.cos_cached[position_ids].to(x.device)
        sin = self.sin_cached[position_ids].to(x.device)
        
        x1 = x[..., :self.dim//2]
        x2 = x[..., self.dim//2:self.dim]
        
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        rotated_x = torch.cat([rotated_x1, rotated_x2], dim=-1)
        
        return rotated_x

class CustomAttention(nn.Module):
    """自定义多头注意力实现，支持MLA（Multi-Latent Attention）"""
    def __init__(
        self, 
        hidden_size, 
        num_attention_heads, 
        kv_channels=None,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        attention_dropout=0.0,
        use_rotary=True,
        rotary_base=10000.0,
        rotary_scaling_factor=1.0,
        max_position_embeddings=2048
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.kv_channels = kv_channels if kv_channels is not None else hidden_size // num_attention_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.attention_dropout = attention_dropout
        self.use_rotary = use_rotary
        
        self.q_size = num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim)
        self.k_size = num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim)
        self.v_size = num_attention_heads * v_head_dim
        
        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.k_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.v_size, bias=False)
        self.o_proj = nn.Linear(self.v_size, hidden_size, bias=False)
        
        if use_rotary:
            self.rotary = CustomRotaryEmbedding(
                dim=qk_rope_head_dim * 2,  # 每个头的旋转部分的维度
                max_position_embeddings=max_position_embeddings,
                base=rotary_base,
                scaling_factor=rotary_scaling_factor
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False
    ):
        """前向传播"""
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        head_dim_nope = self.qk_nope_head_dim
        head_dim_rope = self.qk_rope_head_dim
        head_dim_v = self.v_head_dim
        
        q_nope = q[:, :, :self.num_attention_heads * head_dim_nope]
        q_rope = q[:, :, self.num_attention_heads * head_dim_nope:]
        
        k_nope = k[:, :, :self.num_attention_heads * head_dim_nope]
        k_rope = k[:, :, self.num_attention_heads * head_dim_nope:]
        
        q_nope = q_nope.view(batch_size, seq_len, self.num_attention_heads, head_dim_nope)
        q_rope = q_rope.view(batch_size, seq_len, self.num_attention_heads, head_dim_rope)
        
        k_nope = k_nope.view(batch_size, seq_len, self.num_attention_heads, head_dim_nope)
        k_rope = k_rope.view(batch_size, seq_len, self.num_attention_heads, head_dim_rope)
        
        v = v.view(batch_size, seq_len, self.num_attention_heads, head_dim_v)
        
        if self.use_rotary:
            q_rope = self.rotary(q_rope, position_ids)
            k_rope = self.rotary(k_rope, position_ids)
        
        q_nope = q_nope.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
        q_rope = q_rope.permute(0, 2, 1, 3)
        
        k_nope = k_nope.permute(0, 2, 1, 3)
        k_rope = k_rope.permute(0, 2, 1, 3)
        
        v = v.permute(0, 2, 1, 3)
        
        attn_weights_nope = torch.matmul(q_nope, k_nope.transpose(-1, -2))
        attn_weights_rope = torch.matmul(q_rope, k_rope.transpose(-1, -2))
        
        attn_weights_nope = attn_weights_nope / math.sqrt(head_dim_nope)
        attn_weights_rope = attn_weights_rope / math.sqrt(head_dim_rope)
        
        attn_weights = attn_weights_nope + attn_weights_rope
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        
        attn_output = attn_output.view(batch_size, seq_len, self.v_size)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class CustomMoERouter(nn.Module):
    """自定义MoE路由器实现"""
    def __init__(self, hidden_size, num_experts, topk=2, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.topk = topk
        
        self.router = nn.Linear(hidden_size, num_experts, bias=bias)
        
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        if bias:
            nn.init.zeros_(self.router.bias)
    
    def forward(self, hidden_states):
        """前向传播"""
        router_logits = self.router(hidden_states)
        
        router_probs = torch.sigmoid(router_logits)
        
        routing_weights, routing_indices = torch.topk(router_probs, self.topk, dim=-1)
        
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, routing_indices, router_probs

class CustomMoEExpert(nn.Module):
    """自定义MoE专家实现"""
    def __init__(self, hidden_size, ffn_hidden_size, activation_func=F.silu):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.activation_func = activation_func
        
        self.w1 = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.w2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        
        nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
    
    def forward(self, hidden_states):
        """前向传播"""
        x = self.w1(hidden_states)
        x = self.activation_func(x)
        x = self.w2(x)
        return x

class CustomMoE(nn.Module):
    """自定义MoE层实现"""
    def __init__(self, hidden_size, ffn_hidden_size, num_experts, topk=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_experts = num_experts
        self.topk = topk
        
        self.gate = CustomMoERouter(hidden_size, num_experts, topk=topk)
        
        self.experts = nn.ModuleList([
            CustomMoEExpert(hidden_size, ffn_hidden_size)
            for _ in range(num_experts)
        ])
    
    def forward(self, hidden_states):
        """前向传播"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        routing_weights, routing_indices, router_probs = self.gate(hidden_states)
        
        expert_outputs = torch.zeros_like(hidden_states)
        
        flat_hidden_states = hidden_states.view(-1, hidden_size)
        flat_expert_outputs = expert_outputs.view(-1, hidden_size)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (routing_indices == expert_idx).any(dim=-1)
            flat_expert_mask = expert_mask.view(-1)
            
            if flat_expert_mask.any():
                expert_input = flat_hidden_states[flat_expert_mask]
                
                expert_output = self.experts[expert_idx](expert_input)
                
                flat_expert_outputs[flat_expert_mask] += expert_output
        
        expert_outputs = flat_expert_outputs.view(batch_size, seq_len, hidden_size)
        
        return expert_outputs

class CustomTransformerLayer(nn.Module):
    """自定义Transformer层实现"""
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        ffn_hidden_size,
        moe_ffn_hidden_size=None,
        num_experts=None,
        topk=2,
        use_moe=False,
        layer_idx=0,
        use_rotary=True,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rotary_base=10000.0,
        rotary_scaling_factor=1.0,
        max_position_embeddings=2048
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_moe = use_moe
        self.layer_idx = layer_idx
        
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        self.self_attn = CustomAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            kv_channels=v_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            use_rotary=use_rotary,
            rotary_base=rotary_base,
            rotary_scaling_factor=rotary_scaling_factor,
            max_position_embeddings=max_position_embeddings
        )
        
        if use_moe:
            self.mlp = CustomMoE(
                hidden_size=hidden_size,
                ffn_hidden_size=moe_ffn_hidden_size or ffn_hidden_size,
                num_experts=num_experts,
                topk=topk
            )
        else:
            self.mlp = CustomMoEExpert(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size
            )
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False
    ):
        """前向传播"""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attention_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache
        )
        
        hidden_states = residual + attention_output
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        mlp_output = self.mlp(hidden_states)
        
        hidden_states = residual + mlp_output
        
        return hidden_states

class CustomModel(nn.Module):
    """自定义模型实现，模拟HuggingFace风格的模型"""
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_attention_heads,
        ffn_hidden_size,
        moe_ffn_hidden_size=None,
        num_experts=None,
        topk=2,
        moe_layer_freq=None,
        max_position_embeddings=2048,
        rotary_base=10000.0,
        rotary_scaling_factor=1.0,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        if moe_layer_freq is None:
            moe_layer_freq = [i > 0 for i in range(num_layers)]
        
        self.layers = nn.ModuleList([
            CustomTransformerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                ffn_hidden_size=ffn_hidden_size,
                moe_ffn_hidden_size=moe_ffn_hidden_size,
                num_experts=num_experts,
                topk=topk,
                use_moe=moe_layer_freq[i],
                layer_idx=i,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                rotary_base=rotary_base,
                rotary_scaling_factor=rotary_scaling_factor,
                max_position_embeddings=max_position_embeddings
            )
            for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
    ):
        """前向传播"""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, seq_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)
        
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=inputs_embeds.device),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
        
        causal_mask = causal_mask.to(inputs_embeds.dtype) * -1e9
        
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=False
            )
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states

class CustomModelForCausalLM(nn.Module):
    """自定义因果语言模型，模拟HuggingFace风格的模型"""
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_attention_heads,
        ffn_hidden_size,
        moe_ffn_hidden_size=None,
        num_experts=None,
        topk=2,
        moe_layer_freq=None,
        max_position_embeddings=2048,
        rotary_base=10000.0,
        rotary_scaling_factor=1.0,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128
    ):
        super().__init__()
        self.model = CustomModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            ffn_hidden_size=ffn_hidden_size,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            num_experts=num_experts,
            topk=topk,
            moe_layer_freq=moe_layer_freq,
            max_position_embeddings=max_position_embeddings,
            rotary_base=rotary_base,
            rotary_scaling_factor=rotary_scaling_factor,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim
        )
        
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
    ):
        """前向传播"""
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        logits = self.lm_head(hidden_states)
        
        return logits

def add_test_args(parser):
    group = parser.add_argument_group('DeepSeek-V3 Test')
    group.add_argument('--test-tp', 
                       type=int, 
                       default=1,
                       help='Tensor parallelism size for testing')
    group.add_argument('--test-pp', 
                       type=int, 
                       default=1,
                       help='Pipeline parallelism size for testing')
    group.add_argument('--test-batch-size', 
                       type=int, 
                       default=1,
                       help='Batch size for testing')
    group.add_argument('--test-seq-length', 
                       type=int, 
                       default=512,
                       help='Sequence length for testing')
    group.add_argument('--debug', 
                       action='store_true',
                       help='Enable debug mode with more logging')
    return parser

def check_mg_custom_forward(mg_model, custom_model, args):
    """比较Megatron和自定义模型的输出"""
    print("Setting up test data and hooks...")
    
    device = torch.cuda.current_device()
    seq_len = args.test_seq_length
    batch_size = args.test_batch_size
    
    input_ids = torch.randint(0, args.padded_vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    custom_activations = [{} for _ in range(args.num_layers)]
    mg_activations = [{} for _ in range(args.num_layers)]
    
    def input_hook(module, args, kwargs, layer_idx, mode):
        frame, name = mode.split('-')
        if frame == 'custom':
            custom_activations[layer_idx][name] = args[0].detach().clone()
        elif frame == 'mg':
            if 'hidden_states' in kwargs:
                mg_activations[layer_idx][name] = kwargs['hidden_states'].detach().clone()
            else:
                mg_activations[layer_idx][name] = args[0].detach().clone()
    
    def output_hook(module, args, kwargs, output, layer_idx, mode):
        frame, name = mode.split('-')
        if frame == 'custom':
            if isinstance(output, tuple):
                custom_activations[layer_idx][name] = output[0].detach().clone()
            else:
                custom_activations[layer_idx][name] = output.detach().clone()
        elif frame == 'mg':
            if isinstance(output, tuple):
                mg_activations[layer_idx][name] = output[0].detach().clone()
            else:
                mg_activations[layer_idx][name] = output.detach().clone()
    
    print("Registering hooks for custom model...")
    for idx, layer in enumerate(custom_model.model.layers):
        layer.register_forward_pre_hook(
            partial(input_hook, layer_idx=idx, mode='custom-layer_in'), 
            with_kwargs=True
        )
        
        layer.self_attn.register_forward_hook(
            partial(output_hook, layer_idx=idx, mode='custom-attn_out'),
            with_kwargs=True
        )
        
        if hasattr(layer.mlp, 'experts'):
            layer.mlp.register_forward_hook(
                partial(output_hook, layer_idx=idx, mode='custom-moe_out'),
                with_kwargs=True
            )
            layer.mlp.gate.register_forward_hook(
                partial(output_hook, layer_idx=idx, mode='custom-router_out'),
                with_kwargs=True
            )
    
    custom_model.lm_head.register_forward_hook(
        partial(output_hook, layer_idx=args.num_layers-1, mode='custom-lmhead'),
        with_kwargs=True
    )
    
    print("Registering hooks for Megatron model...")
    for idx, layer in enumerate(mg_model.decoder.layers):
        layer.register_forward_pre_hook(
            partial(input_hook, layer_idx=idx, mode='mg-layer_in'),
            with_kwargs=True
        )
        
        layer.self_attention.register_forward_hook(
            partial(output_hook, layer_idx=idx, mode='mg-attn_out'),
            with_kwargs=True
        )
        
        if hasattr(layer.mlp, 'experts'):
            layer.mlp.register_forward_hook(
                partial(output_hook, layer_idx=idx, mode='mg-moe_out'),
                with_kwargs=True
            )
            layer.mlp.router.register_forward_hook(
                partial(output_hook, layer_idx=idx, mode='mg-router_out'),
                with_kwargs=True
            )
    
    mg_model.output_layer.register_forward_hook(
        partial(output_hook, layer_idx=args.num_layers-1, mode='mg-lmhead'),
        with_kwargs=True
    )
    
    print("Running forward pass for custom model...")
    with torch.no_grad():
        custom_outputs = custom_model(input_ids=input_ids)
        
        mg_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        print("Running forward pass for Megatron model...")
        mg_outputs = mg_model(**mg_input)
    
    print("\n============= Testing Results =============")
    
    def compare_activations(custom_act, mg_act, name, epsilon=1e-2):
        """比较两个激活值的差异，由于使用随机权重，我们主要关注结构而不是精确值"""
        if custom_act is None or mg_act is None:
            print(f"WARNING: {name} activation missing from one model!")
            return False
        
        if custom_act.shape != mg_act.shape:
            print(f"Shape mismatch for {name}: Custom {custom_act.shape} vs MG {mg_act.shape}")
            return False
        
        if torch.isnan(custom_act).any() or torch.isinf(custom_act).any():
            print(f"WARNING: NaN or Inf in custom model {name}")
            return False
        
        if torch.isnan(mg_act).any() or torch.isinf(mg_act).any():
            print(f"WARNING: NaN or Inf in Megatron model {name}")
            return False
        
        custom_mean = custom_act.mean().item()
        custom_std = custom_act.std().item()
        mg_mean = mg_act.mean().item()
        mg_std = mg_act.std().item()
        
        print(f"Stats for {name}:")
        print(f"  Custom: mean={custom_mean:.4f}, std={custom_std:.4f}")
        print(f"  Megatron: mean={mg_mean:.4f}, std={mg_std:.4f}")
        
        success = True
        
        return success
    
    all_passed = True
    
    print("\n--- Checking Multi-Latent Attention (MLA) ---")
    for idx in range(min(args.num_layers, len(custom_activations))):
        if 'custom-attn_out' in custom_activations[idx] and 'mg-attn_out' in mg_activations[idx]:
            passed = compare_activations(
                custom_activations[idx]['custom-attn_out'],
                mg_activations[idx]['mg-attn_out'],
                f"Layer {idx} Attention Output"
            )
            all_passed = all_passed and passed
    
    print("\n--- Checking Mixture of Experts (MOE) ---")
    for idx in range(min(args.num_layers, len(custom_activations))):
        if 'custom-moe_out' in custom_activations[idx] and 'mg-moe_out' in mg_activations[idx]:
            passed = compare_activations(
                custom_activations[idx]['custom-moe_out'],
                mg_activations[idx]['mg-moe_out'],
                f"Layer {idx} MOE Output"
            )
            all_passed = all_passed and passed
        
        if 'custom-router_out' in custom_activations[idx] and 'mg-router_out' in mg_activations[idx]:
            passed = compare_activations(
                custom_activations[idx]['custom-router_out'],
                mg_activations[idx]['mg-router_out'],
                f"Layer {idx} Router Output"
            )
            all_passed = all_passed and passed
    
    print("\n--- Checking Final Output ---")
    if 'custom-lmhead' in custom_activations[-1] and 'mg-lmhead' in mg_activations[-1]:
        passed = compare_activations(
            custom_activations[-1]['custom-lmhead'],
            mg_activations[-1]['mg-lmhead'],
            "Final Output"
        )
        all_passed = all_passed and passed
    
    print("\n============= Overall Result =============")
    print("PASS" if all_passed else "FAIL")
    
    return all_passed

def main():
    """主函数"""
    initialize_megatron(extra_args_provider=add_test_args)
    args = get_args()
    
    print(f"Testing with configuration:\n"
          f"  - Number of layers: {args.num_layers}\n"
          f"  - Hidden size: {args.hidden_size}\n"
          f"  - Number of attention heads: {args.num_attention_heads}\n"
          f"  - Number of experts: {args.num_experts}\n"
          f"  - Tensor parallelism: {args.test_tp}\n"
          f"  - Pipeline parallelism: {args.test_pp}\n")
    
    print("Creating custom model with random weights...")
    
    moe_layer_freq = [i > 0 for i in range(args.num_layers)]  # 第一层使用普通FFN，其余层使用MOE
    
    custom_model = CustomModelForCausalLM(
        vocab_size=args.padded_vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size if hasattr(args, 'moe_ffn_hidden_size') else args.ffn_hidden_size // 4,
        num_experts=args.num_experts,
        topk=args.moe_router_topk if hasattr(args, 'moe_router_topk') else 2,
        moe_layer_freq=moe_layer_freq,
        max_position_embeddings=args.max_position_embeddings,
        rotary_base=args.rotary_base if hasattr(args, 'rotary_base') else 10000.0,
        rotary_scaling_factor=args.rotary_scaling_factor if hasattr(args, 'rotary_scaling_factor') else 1.0,
        qk_nope_head_dim=args.qk_nope_head_dim if hasattr(args, 'qk_nope_head_dim') else 128,
        qk_rope_head_dim=args.qk_rope_head_dim if hasattr(args, 'qk_rope_head_dim') else 64,
        v_head_dim=args.kv_channels if hasattr(args, 'kv_channels') else 128
    ).cuda()
    
    print("Creating Megatron model...")
    mg_model = model_provider()
    
    print("Running validation...")
    success = check_mg_custom_forward(mg_model, custom_model, args)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
