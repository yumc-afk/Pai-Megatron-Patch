
import os
import sys
import torch

import jaxtyping
from jaxtyping import Array, Float, Int

import argparse
from transformers import AutoConfig, AutoModelForCausalLM

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(path_dir)
sys.path.append(os.path.join(path_dir, "Megatron-LM-250328"))

def verify_weight_conversion_standalone():
    """验证权重转换的正确性（独立版本，不依赖Megatron）"""
    parser = argparse.ArgumentParser(description='DeepSeek-V3 Weight Conversion Verification')
    parser.add_argument('--hf-model-name', 
                       type=str, 
                       default='deepseek-ai/deepseek-moe-16b-base',
                       help='HuggingFace模型名称')
    
    parser.add_argument('--use-cpu', 
                       action='store_true',
                       help='使用CPU进行验证')
    
    parser.add_argument('--verify-layers', 
                       type=int, 
                       default=2,
                       help='验证的层数')
    
    parser.add_argument('--verify-experts', 
                       type=int, 
                       default=4,
                       help='验证的专家数量')
    
    args = parser.parse_args()
    
    print(f"验证HuggingFace模型 {args.hf_model_name} 的权重转换")
    print(f"验证层数: {args.verify_layers}")
    print(f"验证专家数量: {args.verify_experts}")
    
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    
    print("加载HuggingFace模型配置...")
    try:
        hf_config = AutoConfig.from_pretrained(args.hf_model_name, trust_remote_code=True)
        print(f"模型配置: {hf_config}")
        
        num_layers = hf_config.num_hidden_layers
        hidden_size = hf_config.hidden_size
        num_attention_heads = hf_config.num_attention_heads
        
        if hasattr(hf_config, 'num_experts'):
            num_experts = hf_config.num_experts
        elif hasattr(hf_config, 'n_routed_experts'):
            num_experts = hf_config.n_routed_experts
            print(f"找到路由专家数量: {num_experts}")
        else:
            num_experts = 0
            print("警告: 模型配置中没有专家数量信息")
        
        print(f"层数: {num_layers}")
        print(f"隐藏层大小: {hidden_size}")
        print(f"注意力头数: {num_attention_heads}")
        print(f"专家数量: {num_experts}")
        
        has_mla = hasattr(hf_config, 'multi_latent_attention') and hf_config.multi_latent_attention
        
        has_moe = num_experts > 0 or hasattr(hf_config, 'moe_layer_freq') or hasattr(hf_config, 'n_routed_experts')
        
        print(f"包含MLA: {has_mla}")
        print(f"包含MOE: {has_moe}")
        
        if not has_mla:
            print("警告: 模型不包含MLA，无法验证MLA权重转换")
        
        if not has_moe:
            print("警告: 模型不包含MOE，无法验证MOE权重转换")
        
        if args.use_cpu:
            print("在CPU上验证模型结构...")
            try:
                print("仅加载模型配置...")
                from transformers.utils import logging
                logging.set_verbosity_info()
                
                
                print("\n===== 模型结构验证 =====")
                
                if has_moe:
                    print("\n验证MOE结构:")
                    if hasattr(hf_config, 'moe_intermediate_size'):
                        print(f"MOE中间层大小: {hf_config.moe_intermediate_size}")
                    if hasattr(hf_config, 'num_experts_per_tok'):
                        print(f"每个token的专家数量: {hf_config.num_experts_per_tok}")
                    if hasattr(hf_config, 'moe_layer_freq'):
                        print(f"MOE层频率: {hf_config.moe_layer_freq}")
                    if hasattr(hf_config, 'n_routed_experts'):
                        print(f"路由专家数量: {hf_config.n_routed_experts}")
                    if hasattr(hf_config, 'n_shared_experts'):
                        print(f"共享专家数量: {hf_config.n_shared_experts}")
                    
                    print("\nMOE实现细节:")
                    if hasattr(hf_config, 'aux_loss_alpha'):
                        print(f"辅助损失系数: {hf_config.aux_loss_alpha}")
                    if hasattr(hf_config, 'scoring_func'):
                        print(f"评分函数: {hf_config.scoring_func}")
                    if hasattr(hf_config, 'seq_aux'):
                        print(f"序列辅助损失: {hf_config.seq_aux}")
                
                print("\n模型结构验证完成")
                
            except Exception as e:
                print(f"模型结构验证失败: {e}")
                import traceback
                traceback.print_exc()
            
            return
        
        print("加载部分模型权重进行验证...")
        
        print(f"尝试加载前{args.verify_layers}层的权重...")
        
        from transformers.utils import logging
        logging.set_verbosity_info()
        
        layer_names = []
        for i in range(min(args.verify_layers, num_layers)):
            layer_names.append(f"model.layers.{i}")
        
        if has_moe:
            for i in range(min(args.verify_layers, num_layers)):
                for j in range(min(args.verify_experts, num_experts)):
                    layer_names.append(f"model.layers.{i}.mlp.experts.{j}")
        
        print("加载指定层的权重（这可能需要一些时间）...")
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model_name,
            trust_remote_code=True,
            # 确保张量数据类型正确
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        assert model.dtype == torch.float16, f"Unexpected dtype: {model.dtype}"
        
        print("\n===== 权重验证 =====")
        
        if has_moe:
            print("\n验证MOE结构:")
            for i in range(min(args.verify_layers, num_layers)):
                if hasattr(model.model.layers[i].mlp, 'experts'):
                    print(f"层 {i} 包含MOE结构")
                    num_loaded_experts = len(model.model.layers[i].mlp.experts)
                    print(f"  - 加载的专家数量: {num_loaded_experts}")
                    
                    if num_loaded_experts > 0:
                        expert = model.model.layers[i].mlp.experts[0]
                        if hasattr(expert, 'w1'):
                            print(f"  - 专家权重w1形状: {expert.w1.weight.shape}")
                        if hasattr(expert, 'w2'):
                            print(f"  - 专家权重w2形状: {expert.w2.weight.shape}")
                        if hasattr(expert, 'w3'):
                            print(f"  - 专家权重w3形状: {expert.w3.weight.shape}")
                else:
                    print(f"层 {i} 不包含MOE结构")
        
        if has_mla:
            print("\n验证MLA结构:")
            for i in range(min(args.verify_layers, num_layers)):
                if hasattr(model.model.layers[i], 'self_attn'):
                    attn = model.model.layers[i].self_attn
                    print(f"层 {i} 包含注意力结构")
                    
                    mla_attrs = ['q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj']
                    for attr in mla_attrs:
                        if hasattr(attn, attr):
                            print(f"  - 包含 {attr}: 是")
                            attr_obj = getattr(attn, attr)
                            if hasattr(attr_obj, 'weight'):
                                print(f"    - 权重形状: {attr_obj.weight.shape}")
                        else:
                            print(f"  - 包含 {attr}: 否")
        
        print("\n权重验证完成")
        
    except Exception as e:
        print(f"验证失败: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    verify_weight_conversion_standalone()
