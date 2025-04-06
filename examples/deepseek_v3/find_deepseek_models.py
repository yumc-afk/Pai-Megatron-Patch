
import os
import sys
from huggingface_hub import list_models

def find_deepseek_moe_models():
    """查找DeepSeek MOE模型，特别是16B版本"""
    print("正在搜索DeepSeek MOE模型...")
    
    models = list_models(author='deepseek-ai')
    
    deepseek_moe_models = []
    deepseek_16b_moe_models = []
    
    for model in models:
        model_id = model.id.lower()
        if 'deepseek' in model_id and ('moe' in model_id or 'mixture' in model_id):
            deepseek_moe_models.append(model.id)
            if '16b' in model_id or '16-b' in model_id:
                deepseek_16b_moe_models.append(model.id)
    
    print("\n所有DeepSeek MOE模型:")
    for model in deepseek_moe_models:
        print(f"- {model}")
    
    print("\nDeepSeek 16B MOE模型:")
    for model in deepseek_16b_moe_models:
        print(f"- {model}")
    
    return deepseek_16b_moe_models

if __name__ == "__main__":
    find_deepseek_moe_models()
