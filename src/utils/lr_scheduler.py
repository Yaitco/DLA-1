def warmup_inverse_sqrt(step: int, warmup_steps: int) -> float:
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be positive")
    step = max(1, step)
    return min(step / warmup_steps, (warmup_steps / step) ** 0.5)
