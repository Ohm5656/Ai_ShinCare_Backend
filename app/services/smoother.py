def ema(series: list[float], alpha: float=0.4) -> list[float]:
    if not series: return []
    out = [series[0]]
    for x in series[1:]:
        out.append(alpha*x + (1-alpha)*out[-1])
    return out
