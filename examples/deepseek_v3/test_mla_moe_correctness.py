#
#
#

import os
import sys
import torch
import argparse
from functools import partial

from transformers import AutoConfig, AutoModelForCausalLM

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(path_dir, "examples"))

from deepseek_v3.pretrain_deepseek import model_provider
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron

def add_test_args(parser):
    group = parser.add_argument_group('DeepSeek-V3 Test')
    group.add_argument('--hf-model-path', 
                       type=str, 
                       required=True,
                       help='Path to HuggingFace model')
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

def check_mg_hf_forward(mg_model, hf_model, args):
    """比较Megatron和HuggingFace模型的输出"""
    print("Setting up test data and hooks...")
    
    device = torch.cuda.current_device()
    seq_len = args.test_seq_length
    batch_size = args.test_batch_size
    
    input_ids = torch.randint(0, args.padded_vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    hf_activations = [{} for _ in range(args.num_layers)]
    mg_activations = [{} for _ in range(args.num_layers)]
    
    def input_hook(module, args, kwargs, layer_idx, mode):
        frame, name = mode.split('-')
        if frame == 'hf':
            hf_activations[layer_idx][name] = args[0].detach().clone()
        elif frame == 'mg':
            if 'hidden_states' in kwargs:
                mg_activations[layer_idx][name] = kwargs['hidden_states'].detach().clone()
            else:
                mg_activations[layer_idx][name] = args[0].detach().clone()
    
    def output_hook(module, args, kwargs, output, layer_idx, mode):
        frame, name = mode.split('-')
        if frame == 'hf':
            if isinstance(output, tuple):
                hf_activations[layer_idx][name] = output[0].detach().clone()
            else:
                hf_activations[layer_idx][name] = output.detach().clone()
        elif frame == 'mg':
            if isinstance(output, tuple):
                mg_activations[layer_idx][name] = output[0].detach().clone()
            else:
                mg_activations[layer_idx][name] = output.detach().clone()
    
    print("Registering hooks for HuggingFace model...")
    for idx, layer in enumerate(hf_model.model.layers):
        layer.register_forward_pre_hook(
            partial(input_hook, layer_idx=idx, mode='hf-layer_in'), 
            with_kwargs=True
        )
        
        layer.self_attn.register_forward_hook(
            partial(output_hook, layer_idx=idx, mode='hf-attn_out'),
            with_kwargs=True
        )
        
        if hasattr(layer.mlp, 'experts'):
            layer.mlp.register_forward_hook(
                partial(output_hook, layer_idx=idx, mode='hf-moe_out'),
                with_kwargs=True
            )
            layer.mlp.gate.register_forward_hook(
                partial(output_hook, layer_idx=idx, mode='hf-router_out'),
                with_kwargs=True
            )
    
    hf_model.lm_head.register_forward_hook(
        partial(output_hook, layer_idx=args.num_layers-1, mode='hf-lmhead'),
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
    
    print("Running forward pass for HuggingFace model...")
    with torch.no_grad():
        hf_outputs = hf_model(input_ids=input_ids)
        
        mg_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        print("Running forward pass for Megatron model...")
        mg_outputs = mg_model(**mg_input)
    
    print("\n============= Testing Results =============")
    
    def compare_activations(hf_act, mg_act, name, epsilon=1e-5):
        if hf_act is None or mg_act is None:
            print(f"WARNING: {name} activation missing from one model!")
            return False
        
        if hf_act.shape != mg_act.shape:
            print(f"Shape mismatch for {name}: HF {hf_act.shape} vs MG {mg_act.shape}")
            return False
        
        diff = (hf_act - mg_act).abs()
        max_diff = diff.max().item()
        avg_diff = diff.mean().item()
        
        success = max_diff < epsilon
        status = "PASS" if success else "FAIL"
        print(f"{status}: {name} - Max diff: {max_diff:.6f}, Avg diff: {avg_diff:.6f}")
        
        return success
    
    all_passed = True
    
    print("\n--- Checking Multi-Latent Attention (MLA) ---")
    for idx in range(min(args.num_layers, len(hf_activations))):
        if 'hf-attn_out' in hf_activations[idx] and 'mg-attn_out' in mg_activations[idx]:
            passed = compare_activations(
                hf_activations[idx]['hf-attn_out'],
                mg_activations[idx]['mg-attn_out'],
                f"Layer {idx} Attention Output"
            )
            all_passed = all_passed and passed
    
    print("\n--- Checking Mixture of Experts (MOE) ---")
    for idx in range(min(args.num_layers, len(hf_activations))):
        if 'hf-moe_out' in hf_activations[idx] and 'mg-moe_out' in mg_activations[idx]:
            passed = compare_activations(
                hf_activations[idx]['hf-moe_out'],
                mg_activations[idx]['mg-moe_out'],
                f"Layer {idx} MOE Output"
            )
            all_passed = all_passed and passed
        
        if 'hf-router_out' in hf_activations[idx] and 'mg-router_out' in mg_activations[idx]:
            passed = compare_activations(
                hf_activations[idx]['hf-router_out'],
                mg_activations[idx]['mg-router_out'],
                f"Layer {idx} Router Output"
            )
            all_passed = all_passed and passed
    
    print("\n--- Checking Final Output ---")
    if 'hf-lmhead' in hf_activations[-1] and 'mg-lmhead' in mg_activations[-1]:
        passed = compare_activations(
            hf_activations[-1]['hf-lmhead'],
            mg_activations[-1]['mg-lmhead'],
            "Final Output"
        )
        all_passed = all_passed and passed
    
    print("\n============= Overall Result =============")
    print("PASS" if all_passed else "FAIL")
    
    return all_passed

def main():
    initialize_megatron(extra_args_provider=add_test_args)
    args = get_args()
    
    print(f"Testing with configuration:\n"
          f"  - Number of layers: {args.num_layers}\n"
          f"  - Hidden size: {args.hidden_size}\n"
          f"  - Number of attention heads: {args.num_attention_heads}\n"
          f"  - Number of experts: {args.num_experts}\n"
          f"  - Tensor parallelism: {args.test_tp}\n"
          f"  - Pipeline parallelism: {args.test_pp}\n")
    
    print(f"Loading HuggingFace model from {args.hf_model_path}...")
    hf_config = AutoConfig.from_pretrained(args.hf_model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).cuda()
    
    print("Creating Megatron model...")
    mg_model = model_provider()
    
    print("Running validation...")
    success = check_mg_hf_forward(mg_model, hf_model, args)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
