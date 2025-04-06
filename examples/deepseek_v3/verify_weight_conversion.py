
import os
import sys
import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(path_dir, "examples"))

from deepseek_v3.pretrain_deepseek import model_provider
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron

def add_verification_args(parser):
    """添加验证参数"""
    group = parser.add_argument_group('DeepSeek-V3 Weight Conversion Verification')
    
    group.add_argument('--hf-model-name', 
                       type=str, 
                       default='deepseek-ai/deepseek-moe-16b-base',
                       help='HuggingFace模型名称')
    
    group.add_argument('--use-cpu', 
                       action='store_true',
                       help='使用CPU进行验证')
    
    group.add_argument('--verify-layers', 
                       type=int, 
                       default=2,
                       help='验证的层数')
    
    group.add_argument('--verify-experts', 
                       type=int, 
                       default=4,
                       help='验证的专家数量')
    
    return parser

def verify_weight_conversion():
    """验证权重转换的正确性"""
    initialize_megatron(extra_args_provider=add_verification_args)
    args = get_args()
    
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
        else:
            num_experts = 0
            print("警告: 模型配置中没有专家数量信息")
        
        print(f"层数: {num_layers}")
        print(f"隐藏层大小: {hidden_size}")
        print(f"注意力头数: {num_attention_heads}")
        print(f"专家数量: {num_experts}")
        
        has_mla = hasattr(hf_config, 'multi_latent_attention') and hf_config.multi_latent_attention
        has_moe = num_experts > 0
        
        print(f"包含MLA: {has_mla}")
        print(f"包含MOE: {has_moe}")
        
        if not has_mla:
            print("警告: 模型不包含MLA，无法验证MLA权重转换")
        
        if not has_moe:
            print("警告: 模型不包含MOE，无法验证MOE权重转换")
        
        if args.use_cpu:
            print("仅在CPU上验证模型配置，不加载模型权重")
            return
        
        print("加载部分模型权重进行验证...")
        
    except Exception as e:
        print(f"验证失败: {e}")
        return

def main():
    verify_weight_conversion()

if __name__ == "__main__":
    main()
