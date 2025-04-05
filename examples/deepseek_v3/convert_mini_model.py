#
#
#

import os
import sys
import torch
import argparse
import re
from collections import defaultdict
from transformers import AutoConfig, AutoModelForCausalLM

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(path_dir, "examples"))
from deepseek_v3.pretrain_deepseek import model_provider
from megatron_patch.arguments import get_patch_args
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata

def add_model_args(parser):
    group = parser.add_argument_group('DeepSeek-V3 Mini Test')
    
    group.add_argument('--hf-model-path', 
                       type=str, 
                       required=True,
                       help='Path to HuggingFace model')
    
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

def convert_and_save_mini_model():
    """转换HF模型到小型Megatron模型用于测试"""
    initialize_megatron(extra_args_provider=add_model_args)
    args = get_args()
    
    print(f"Converting HuggingFace model from {args.hf_model_path}")
    print(f"Creating mini model with {args.num_target_layers} layers")
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print("Loading HuggingFace model...")
    hf_config = AutoConfig.from_pretrained(args.hf_model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path,
        trust_remote_code=True,
    )
    
    print("Creating Megatron model...")
    mg_model = model_provider()
    
    print("Converting weights...")
    
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
    args.expert_model_parallel_size = args.target_expert_model_parallel_size
    args.expert_tensor_parallel_size = args.target_expert_tensor_parallel_size
    
    num_pp_layers = [args.num_target_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size
    for i in range(args.num_target_layers % args.pipeline_model_parallel_size):
        num_pp_layers[i] += 1
    
    print(f"Pipeline parallelism layer distribution: {num_pp_layers}")
    
    state_dict = {}
    
    print("Converting embedding layer...")
    state_dict['embedding.word_embeddings.weight'] = hf_model.model.embed_tokens.weight.clone()
    
    print("Converting transformer layers...")
    for dest_layer_idx in range(args.num_target_layers):
        src_layer_idx = dest_layer_idx  # 从头开始选择层
        
        print(f"Converting layer {src_layer_idx} to layer {dest_layer_idx}")
        
        pp_rank = 0
        pp_layer_idx = dest_layer_idx
        
        for rank, num_layers in enumerate(num_pp_layers):
            if pp_layer_idx < num_layers:
                pp_rank = rank
                break
            pp_layer_idx -= num_layers
        
        prefix = f"decoder.layers.{dest_layer_idx}."
        state_dict[prefix + 'input_layernorm.weight'] = hf_model.model.layers[src_layer_idx].input_layernorm.weight.clone()
        state_dict[prefix + 'pre_mlp_layernorm.weight'] = hf_model.model.layers[src_layer_idx].post_attention_layernorm.weight.clone()
        
        prefix_attn = prefix + 'self_attention.'
        
        state_dict[prefix_attn + 'query_key_value.weight'] = hf_model.model.layers[src_layer_idx].self_attn.q_proj.weight.clone()
        state_dict[prefix_attn + 'query_key_value.bias'] = torch.zeros_like(hf_model.model.layers[src_layer_idx].self_attn.q_proj.weight[0])
        
        state_dict[prefix_attn + 'dense.weight'] = hf_model.model.layers[src_layer_idx].self_attn.o_proj.weight.clone()
        
        if hasattr(hf_model.model.layers[src_layer_idx].mlp, 'experts'):
            prefix_mlp = prefix + 'mlp.'
            
            state_dict[prefix_mlp + 'router.weight'] = hf_model.model.layers[src_layer_idx].mlp.gate.weight.clone()
            
            for expert_idx in range(args.num_experts):
                if expert_idx < len(hf_model.model.layers[src_layer_idx].mlp.experts):
                    expert = hf_model.model.layers[src_layer_idx].mlp.experts[expert_idx]
                    
                    state_dict[prefix_mlp + f'experts.{expert_idx}.fc1.weight'] = expert.w1.weight.clone()
                    
                    state_dict[prefix_mlp + f'experts.{expert_idx}.fc2.weight'] = expert.w2.weight.clone()
        else:
            prefix_mlp = prefix + 'mlp.'
            state_dict[prefix_mlp + 'fc1.weight'] = hf_model.model.layers[src_layer_idx].mlp.up_proj.weight.clone()
            state_dict[prefix_mlp + 'fc2.weight'] = hf_model.model.layers[src_layer_idx].mlp.down_proj.weight.clone()
    
    state_dict['decoder.final_layernorm.weight'] = hf_model.model.norm.weight.clone()
    state_dict['output_layer.weight'] = hf_model.lm_head.weight.clone()
    
    print(f"Saving converted model to {args.save_path}...")
    
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
    
    print("Model conversion completed successfully!")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    args = parser.parse_args()
    
    return convert_and_save_mini_model()

if __name__ == "__main__":
    sys.exit(main())
