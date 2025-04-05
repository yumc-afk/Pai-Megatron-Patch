#
#
#

import os
import sys
import torch
import argparse
import math
from collections import defaultdict

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(path_dir, "examples"))
from deepseek_v3.pretrain_deepseek import model_provider
from megatron_patch.arguments import get_patch_args
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata

def add_model_args(parser):
    group = parser.add_argument_group('DeepSeek-V3 Mini Test')
    
    group.add_argument('--target-tensor-model-parallel-size',
                       type=int,
                       default=1,
                       help='Target tensor parallel size')
    
    group.add_argument('--target-pipeline-model-parallel-size',
                       type=int,
                       default=1,
                       help='Target pipeline parallel size')
    
    group.add_argument('--target-expert-model-parallel-size',
                       type=int,
                       default=1,
                       help='Target expert model parallel size')
    
    group.add_argument('--target-expert-tensor-parallel-size',
                       type=int,
                       default=1,
                       help='Target expert tensor parallel size')
    
    group.add_argument('--save-path',
                       type=str,
                       default='./mini_model_ckpt',
                       help='Path to save converted model')
    
    group.add_argument('--num-target-layers',
                       type=int,
                       default=6,
                       help='Number of layers to convert')
    
    return parser

def create_random_mini_model():
    """创建随机初始化的小型Megatron模型用于测试"""
    initialize_megatron(extra_args_provider=add_model_args)
    args = get_args()
    
    print(f"Creating mini model with {args.num_target_layers} layers")
    print(f"Configuration: hidden_size={args.hidden_size}, num_experts={args.num_experts}")
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print("Creating Megatron model with random weights...")
    
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
    args.expert_model_parallel_size = args.target_expert_model_parallel_size
    args.expert_tensor_parallel_size = args.target_expert_tensor_parallel_size
    
    num_pp_layers = [args.num_target_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size
    for i in range(args.num_target_layers % args.pipeline_model_parallel_size):
        num_pp_layers[i] += 1
    
    print(f"Pipeline parallelism layer distribution: {num_pp_layers}")
    
    state_dict = {}
    
    print("Creating random embedding layer...")
    state_dict['embedding.word_embeddings.weight'] = torch.randn(
        args.padded_vocab_size, args.hidden_size
    ) * 0.02
    
    print("Creating random transformer layers...")
    for layer_idx in range(args.num_target_layers):
        print(f"Creating layer {layer_idx}")
        
        pp_rank = 0
        pp_layer_idx = layer_idx
        
        for rank, num_layers in enumerate(num_pp_layers):
            if pp_layer_idx < num_layers:
                pp_rank = rank
                break
            pp_layer_idx -= num_layers
        
        prefix = f"decoder.layers.{layer_idx}."
        state_dict[prefix + 'input_layernorm.weight'] = torch.ones(args.hidden_size)
        state_dict[prefix + 'pre_mlp_layernorm.weight'] = torch.ones(args.hidden_size)
        
        prefix_attn = prefix + 'self_attention.'
        
        qkv_size = args.num_attention_heads * args.kv_channels
        state_dict[prefix_attn + 'query_key_value.weight'] = torch.randn(
            qkv_size, args.hidden_size
        ) * math.sqrt(2.0 / (5 * args.hidden_size))
        state_dict[prefix_attn + 'query_key_value.bias'] = torch.zeros(qkv_size)
        
        state_dict[prefix_attn + 'dense.weight'] = torch.randn(
            args.hidden_size, qkv_size
        ) * math.sqrt(2.0 / (args.hidden_size + qkv_size))
        
        use_moe = layer_idx > 0  # 第一层使用普通FFN，其余层使用MOE
        
        if use_moe:
            prefix_mlp = prefix + 'mlp.'
            
            state_dict[prefix_mlp + 'router.weight'] = torch.randn(
                args.num_experts, args.hidden_size
            ) * 0.01
            
            for expert_idx in range(args.num_experts):
                state_dict[prefix_mlp + f'experts.{expert_idx}.fc1.weight'] = torch.randn(
                    args.ffn_hidden_size, args.hidden_size
                ) * math.sqrt(2.0 / (args.hidden_size + args.ffn_hidden_size))
                
                state_dict[prefix_mlp + f'experts.{expert_idx}.fc2.weight'] = torch.randn(
                    args.hidden_size, args.ffn_hidden_size
                ) * math.sqrt(2.0 / (args.hidden_size + args.ffn_hidden_size))
        else:
            prefix_mlp = prefix + 'mlp.'
            state_dict[prefix_mlp + 'fc1.weight'] = torch.randn(
                args.ffn_hidden_size, args.hidden_size
            ) * math.sqrt(2.0 / (args.hidden_size + args.ffn_hidden_size))
            
            state_dict[prefix_mlp + 'fc2.weight'] = torch.randn(
                args.hidden_size, args.ffn_hidden_size
            ) * math.sqrt(2.0 / (args.hidden_size + args.ffn_hidden_size))
    
    state_dict['decoder.final_layernorm.weight'] = torch.ones(args.hidden_size)
    state_dict['output_layer.weight'] = torch.randn(
        args.padded_vocab_size, args.hidden_size
    ) * 0.02
    
    print(f"Saving random model to {args.save_path}...")
    
    for tp_rank in range(args.tensor_model_parallel_size):
        for pp_rank in range(args.pipeline_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                for etp_rank in range(args.expert_tensor_parallel_size):
                    checkpoint_dir = os.path.join(
                        args.save_path,
                        f'tp_{tp_rank:02d}',
                        f'pp_{pp_rank:02d}',
                        f'ep_{ep_rank:02d}',
                        f'etp_{etp_rank:02d}'
                    )
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    checkpoint_path = os.path.join(checkpoint_dir, 'mp_rank_00_model_states.pt')
                    
                    torch.save({
                        'module': state_dict,
                        'optimizer': None,
                        'lr_scheduler': None,
                        'iteration': 0,
                        'args': args
                    }, checkpoint_path)
    
    tracker_filename = get_checkpoint_tracker_filename(args.save_path)
    with open(tracker_filename, 'w') as f:
        f.write('0\n')
    
    print("Random model creation completed successfully!")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    args = parser.parse_args()
    
    return create_random_mini_model()

if __name__ == "__main__":
    sys.exit(main())
