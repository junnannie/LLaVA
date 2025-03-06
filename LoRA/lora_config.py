

# LoRA 配置
class LoraConfig:
    """
    SigLip 的投影矩阵叫做, out_proj
    Qwen 的投影矩阵叫做, o_proj
    """
    rank = 8
    alpha = 16
    # target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj']
