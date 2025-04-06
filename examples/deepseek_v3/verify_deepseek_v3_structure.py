
import os
import sys
import torch

import jaxtyping
from jaxtyping import Array, Float, Int

import argparse
from transformers import AutoConfig, AutoModelForCausalLM

def verify_model_structure():
    """验证DeepSeek V3模型结构，特别关注MLA和MOE组件"""
    parser = argparse.ArgumentParser(description='验证DeepSeek V3模型结构')
    parser.add_argument('--model-name', type=str, default='deepseek-ai/deepseek-v3-7b',
                        help='HuggingFace模型名称')
    parser.add_argument('--config-only', action='store_true',
                        help='仅验证模型配置，不加载权重')
    args = parser.parse_args()
    
    print(f"验证模型: {args.model_name}")
    
    try:
        print("加载模型配置...")
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        
        print("\n===== 模型配置 =====")
        print(f"模型类型: {config.model_type}")
        print(f"隐藏层大小: {config.hidden_size}")
        print(f"层数: {config.num_hidden_layers}")
        print(f"注意力头数: {config.num_attention_heads}")
        
        has_mla = False
        mla_attrs = ['multi_latent_attention', 'qk_head_dim', 'qk_pos_emb_head_dim', 'v_head_dim']
        for attr in mla_attrs:
            if hasattr(config, attr):
                has_mla = True
                print(f"{attr}: {getattr(config, attr)}")
        
        if has_mla:
            print("模型包含MLA (Multi-Latent Attention)")
        else:
            print("模型不包含MLA (Multi-Latent Attention)")
        
        has_moe = False
        if hasattr(config, 'num_experts') and config.num_experts > 0:
            has_moe = True
            print(f"专家数量: {config.num_experts}")
            if hasattr(config, 'router_topk'):
                print(f"路由器TopK: {config.router_topk}")
            if hasattr(config, 'moe_layer_freq'):
                print(f"MOE层频率: {config.moe_layer_freq}")
        
        if has_moe:
            print("模型包含MOE (Mixture of Experts)")
        else:
            print("模型不包含MOE (Mixture of Experts)")
        
        if args.config_only:
            print("\n仅验证模型配置完成")
            return
        
        print("\n加载模型权重（这可能需要一些时间）...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            trust_remote_code=True,
            # 确保张量数据类型正确
            assert torch_dtype.dtype, f"Unexpected dtype: {{tensor.dtype}}"
            torch_dtype=torch.float16,  # 使用半精度减少内存使用
            device_map="auto"  # 自动处理设备映射
        )
        
        print("\n===== 模型结构验证 =====")
        
        if has_mla:
            print("\n验证MLA结构:")
            if hasattr(model.model.layers[0], 'self_attn'):
                attn = model.model.layers[0].self_attn
                print(f"注意力类型: {type(attn).__name__}")
                
                mla_attrs = ['q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj']
                for attr in mla_attrs:
                    if hasattr(attn, attr):
                        print(f"  - 包含 {attr}: 是")
                    else:
                        print(f"  - 包含 {attr}: 否")
        
        if has_moe:
            print("\n验证MOE结构:")
            moe_layer_idx = None
            for i, layer in enumerate(model.model.layers):
                if hasattr(layer.mlp, 'experts'):
                    moe_layer_idx = i
                    break
            
            if moe_layer_idx is not None:
                moe = model.model.layers[moe_layer_idx].mlp
                print(f"MOE层索引: {moe_layer_idx}")
                print(f"MOE类型: {type(moe).__name__}")
                
                if hasattr(moe, 'experts'):
                    num_experts = len(moe.experts)
                    print(f"专家数量: {num_experts}")
                    print(f"专家类型: {type(moe.experts[0]).__name__}")
                
                if hasattr(moe, 'gate'):
                    print(f"路由器类型: {type(moe.gate).__name__}")
                    if hasattr(moe.gate, 'weight'):
                        print(f"路由器权重形状: {moe.gate.weight.shape}")
            else:
                print("未找到MOE层")
        
        print("\n模型结构验证完成")
        
    except Exception as e:
        print(f"验证失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_model_structure()
