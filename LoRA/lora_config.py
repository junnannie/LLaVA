

class LoraConfig:
    rank = 8
    alpha = 16
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
