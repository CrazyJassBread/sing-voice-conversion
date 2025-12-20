import torch
import random
from typing import List, Tuple, Optional, Dict, Union

# ----------------- 工具：连续有声 run 与不重叠采样 -----------------
def _voiced_runs(mask: torch.Tensor) -> List[Tuple[int, int]]:
    """
    mask: [T] bool tensor
    返回 [(start, end)]，闭区间，均为全局帧索引。
    """
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return []
    # 找到断点（相邻索引差 != 1）
    diff = idx[1:] - idx[:-1]
    cuts = torch.nonzero(diff != 1, as_tuple=False).squeeze(1)
    starts = torch.cat([idx[:1], idx[cuts + 1]])
    ends   = torch.cat([idx[cuts], idx[-1:]])
    return [(int(s.item()), int(e.item())) for s, e in zip(starts, ends)]

def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])

def _sample_nonoverlap_segment(
    runs: List[Tuple[int,int]],
    min_len: int,
    max_len: int,
    existing: List[Tuple[int,int]],
    fixed_len: Optional[int] = None,
    max_tries: int = 5
) -> Optional[Tuple[int,int]]:
    """
    在给定 runs 中采样一个与 existing 不重叠的段。
    若 fixed_len 给定则使用固定长度，否则在 [min_len, max_len] 随机。
    失败返回 None。
    """
    if not runs:
        return None
    tries = 0
    while tries < max_tries:
        need = fixed_len if fixed_len is not None else min_len
        cand = [r for r in runs if (r[1] - r[0] + 1) >= need]
        if not cand:
            return None
        rs, re = random.choice(cand)
        run_len = re - rs + 1
        L = fixed_len if fixed_len is not None else random.randint(min_len, min(max_len, run_len))
        max_start = re - L + 1
        if max_start < rs:
            tries += 1
            continue
        start = random.randint(rs, max_start)
        seg = (start, start + L - 1)
        if all(not _overlap(seg, ex) for ex in existing):
            return seg
        tries += 1
    return None

# ----------------- 工具：Hz ↔ Cents -----------------
def _hz_to_cents(hz: torch.Tensor, ref_hz: float, eps: float = 1e-12) -> torch.Tensor:
    return 1200.0 * torch.log2(torch.clamp(hz, min=eps) / ref_hz)

def _cents_to_hz(cents: torch.Tensor, ref_hz: float) -> torch.Tensor:
    return ref_hz * torch.pow(2.0, cents / 1200.0)

# ----------------- 主函数（纯 tensor 版） -----------------
def apply_f0_perturbations_cent_with_meta(
    f0: torch.Tensor,                                # [T] 或 [B, T]，单位 Hz
    # 概率 & 统一段数范围
    jitter_prob: float = 0.1,
    glide_prob:  float = 0.1,
    jump_prob:   float = 0.3,
    min_segments: int = 2,
    max_segments: int = 4,
    # jitter（按比例映射为 cent 幅度）
    jitter_strength: float = 0.05,
    min_jitter_segment: int = 5,
    max_jitter_segment: int = 20,
    # glide（按比例映射为每帧 cent 斜率）
    glide_max_slope: float = 0.02,
    min_glide_segment: int = 10,
    max_glide_segment: int = 20,
    # jump（段长范围 + 乐理间隔 + 微扰，均在 cent 域）
    sample_rate_hz: int = 100,                       # F0 帧率（帧/秒）
    jump_duration_ms: int = 100,                     # 若未给范围，则用这个固定长度
    jump_min_duration_ms: Optional[int] = 50,      # 段长下限（毫秒）
    jump_max_duration_ms: Optional[int] = 200,      # 段长上限（毫秒）
    jump_base_intervals_cents: Tuple[float, ...] = (100.0, 300.0, 400.0, 1200.0),
    jump_extra_cents_max: float = 50.0,              # 0..max 的额外上偏
    jump_allow_downward: bool = True,                # 允许向下跳变（取负号）
    # 静音 & 参考
    silence_threshold: float = 1e-5,
    ref_hz_for_cents: float = 10.0,
    # 可选：强制模式 'jitter'/'glide'/'jump'/'none'
    force_mode: Optional[str] = None,
) -> Tuple[torch.Tensor, Union[List[Dict], List[List[Dict]]]]:
    """
    返回:
      - f0_out: 与 f0 同形状/同 dtype/同 device
      - meta: 若输入 [T] → List[Dict]；若输入 [B, T] → List[List[Dict]]（每条样本一个列表）

    变换全部在 cent 域完成；选段仅在连续有声 run 内，且不与已选段重叠；不满足最多重选 5 次。
    """
    # ---- 检查与准备 ----
    assert 0.0 <= jitter_prob <= 1.0 and 0.0 <= glide_prob <= 1.0 and 0.0 <= jump_prob <= 1.0
    assert jitter_prob + glide_prob + jump_prob <= 1.0
    assert 1 <= min_segments <= max_segments

    device = f0.device
    dtype  = f0.dtype

    # 统一成 [B, T]
    squeeze_back = False
    if f0.dim() == 1:
        f0 = f0.unsqueeze(0)
        squeeze_back = True
    elif f0.dim() != 2:
        raise ValueError("f0 must be 1D [T] or 2D [B, T] tensor")

    B, T = f0.shape
    out = f0.clone()

    # jump 段长范围（毫秒→帧）
    frames_per_ms = sample_rate_hz / 1000.0
    if jump_min_duration_ms is None:
        jump_min_duration_ms = jump_duration_ms
    if jump_max_duration_ms is None:
        jump_max_duration_ms = jump_duration_ms
    if jump_min_duration_ms > jump_max_duration_ms:
        jump_min_duration_ms, jump_max_duration_ms = jump_max_duration_ms, jump_min_duration_ms
    jump_min_len = max(1, int(round(jump_min_duration_ms * frames_per_ms)))
    jump_max_len = max(1, int(round(jump_max_duration_ms * frames_per_ms)))

    # 比例→cent 的映射
    # jitter：± jitter_cents_amp
    jitter_cents_amp = (1200.0 * torch.log2(torch.tensor(1.0 + jitter_strength, device=device, dtype=torch.float32))).item()

    def _rand_slope_cents(max_ratio: float) -> float:
        r = random.random() * max_ratio  # 0..max_ratio
        return (1200.0 * torch.log2(torch.tensor(1.0 + r, device=device, dtype=torch.float32))).item()

    # ------- 主循环 -------
    all_meta: List[List[Dict]] = []

    for i in range(B):
        x = f0[i]  # [T]
        # 有声帧
        voiced_mask = torch.isfinite(x) & (x > silence_threshold)
        runs = _voiced_runs(voiced_mask)
        sample_meta: List[Dict] = []

        if not runs:
            all_meta.append(sample_meta)
            continue

        # 选择模式
        if force_mode is not None:
            mode = force_mode
        else:
            r = random.random()
            if r < jitter_prob:
                mode = 'jitter'
            elif r < jitter_prob + glide_prob:
                mode = 'glide'
            elif r < jitter_prob + glide_prob + jump_prob:
                mode = 'jump'
            else:
                mode = 'none'

        n_seg = random.randint(min_segments, max_segments)
        chosen: List[Tuple[int,int]] = []

        if mode == 'jitter':
            for _ in range(n_seg):
                seg = _sample_nonoverlap_segment(runs, min_jitter_segment, max_jitter_segment, chosen, None, 5)
                if seg is None:
                    continue
                s, e = seg
                seg_hz = out[i, s:e+1]
                seg_c  = _hz_to_cents(seg_hz, ref_hz_for_cents)
                noise  = (torch.rand_like(seg_c) * 2.0 - 1.0) * jitter_cents_amp
                seg_c_new = seg_c + noise
                out[i, s:e+1] = _cents_to_hz(seg_c_new, ref_hz_for_cents).clamp(min=0.0)
                chosen.append(seg)
                sample_meta.append({
                    'type': 'jitter',
                    'segment': (s, e),
                    'extra': {'amp_cents': jitter_cents_amp, 'len_frames': int(e - s + 1)}
                })

        elif mode == 'glide':
            for _ in range(n_seg):
                seg = _sample_nonoverlap_segment(runs, min_glide_segment, max_glide_segment, chosen, None, 5)
                if seg is None:
                    continue
                s, e = seg
                L = e - s + 1
                seg_hz = out[i, s:e+1]
                seg_c  = _hz_to_cents(seg_hz, ref_hz_for_cents)
                slope_cents = _rand_slope_cents(glide_max_slope)
                if random.random() < 0.5:
                    slope_cents = -slope_cents
                ramp = torch.arange(L, device=device, dtype=seg_c.dtype) * slope_cents
                seg_c_new = seg_c + ramp
                out[i, s:e+1] = _cents_to_hz(seg_c_new, ref_hz_for_cents).clamp(min=0.0)
                chosen.append(seg)
                sample_meta.append({
                    'type': 'glide',
                    'segment': (s, e),
                    'extra': {'slope_cents_per_frame': float(slope_cents), 'len_frames': int(L)}
                })

        elif mode == 'jump':
            for _ in range(n_seg):
                seg = _sample_nonoverlap_segment(runs, jump_min_len, jump_max_len, chosen, None, 5)
                if seg is None:
                    continue
                s, e = seg
                L = e - s + 1
                seg_hz = out[i, s:e+1]
                seg_c  = _hz_to_cents(seg_hz, ref_hz_for_cents)

                base = random.choice(jump_base_intervals_cents) if len(jump_base_intervals_cents) > 0 else 1200.0
                extra = random.random() * max(0.0, jump_extra_cents_max)  # 0..max
                delta_cents = base + extra
                if jump_allow_downward and random.random() < 0.5:
                    delta_cents = -delta_cents

                seg_c_new = seg_c + delta_cents
                out[i, s:e+1] = _cents_to_hz(seg_c_new, ref_hz_for_cents).clamp(min=0.0)
                chosen.append(seg)
                sample_meta.append({
                    'type': 'jump',
                    'segment': (s, e),
                    'extra': {'delta_cents': float(delta_cents), 'base_cents': float(base), 'len_frames': int(L)}
                })

        else:
            # none：不做增强
            pass

        all_meta.append(sample_meta)

    # 还原形状
    if squeeze_back:
        out = out.squeeze(0)
        return out, all_meta[0]
    else:
        return out, all_meta

##############################################
# 检测探针
import torch
from typing import List, Tuple, Optional, Dict, Union

def _contiguous_true_runs(mask_1d: torch.Tensor) -> List[Tuple[int, int]]:
    """给 [T] 的 bool mask，返回所有 True 的连续区间 [(s,e)]，闭区间。"""
    idx = torch.nonzero(mask_1d, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return []
    diff = idx[1:] - idx[:-1]
    cuts = torch.nonzero(diff != 1, as_tuple=False).squeeze(1)
    starts = torch.cat([idx[:1], idx[cuts + 1]])
    ends   = torch.cat([idx[cuts], idx[-1:]])
    return [(int(s.item()), int(e.item())) for s, e in zip(starts, ends)]

def _safe_delta_cents(f0_old: torch.Tensor, f0_new: torch.Tensor, ref_hz: float, silence_th: float) -> torch.Tensor:
    """仅在 voiced（两者都>阈值且有限）处计算 Δcents；其他位置返回 0。"""
    voiced = torch.isfinite(f0_old) & torch.isfinite(f0_new) & (f0_old > silence_th) & (f0_new > silence_th)
    cents_old = torch.zeros_like(f0_old, dtype=torch.float32)
    cents_new = torch.zeros_like(f0_new, dtype=torch.float32)
    cents_old[voiced] = 1200.0 * torch.log2(f0_old[voiced] / ref_hz)
    cents_new[voiced] = 1200.0 * torch.log2(f0_new[voiced] / ref_hz)
    delta = cents_new - cents_old
    delta[~voiced] = 0.0
    return delta, voiced

def verify_f0_change(
    f0_old: torch.Tensor,              # [T] 或 [B,T]，Hz
    f0_new: torch.Tensor,              # 同形状
    *,
    silence_threshold: float = 1e-5,
    atol_hz: float = 1e-6,             # 判定变化的绝对阈值（Hz）
    ref_hz_for_cents: float = 10.0,
    sample_rate_hz: Optional[float] = None,  # 若提供，将给出片段起止时间（秒）
    glide_r2_min: float = 0.8,         # 将片段判为“glide-like”的最小 R^2
    jump_std_max_cents: float = 8.0,   # 将片段判为“jump-like”的 Δcents 标准差上限
) -> Dict[str, Union[bool, dict, list]]:
    """
    返回:
    {
      'has_change': bool,
      'overall': {...},
      'per_sample': [
         {'idx': 0, 'changed_frames': ..., 'ratio': ..., 'abs_hz_mean': ..., 'abs_hz_max': ...,
          'abs_cents_mean_voiced': ..., 'abs_cents_max_voiced': ...,
          'segments': [
             {'start': s, 'end': e, 'len': L, 'start_time': t0, 'end_time': t1,
              'delta_hz_mean': ..., 'delta_hz_max': ...,
              'delta_cents_mean_voiced': ..., 'delta_cents_std_voiced': ...,
              'glide_slope_cents_per_frame': ..., 'glide_slope_cents_per_sec': ...,
              'glide_r2': ..., 'type': 'jump-like'|'glide-like'|'jitter-like'|'unvoiced-or-mixed'}
          ]
         }, ...
      ]
    }
    """
    if f0_old.shape != f0_new.shape:
        raise ValueError("f0_old and f0_new must have the same shape")
    if f0_old.dim() == 1:
        f0_old = f0_old.unsqueeze(0)
        f0_new = f0_new.unsqueeze(0)
        squeeze_back = True
    elif f0_old.dim() == 2:
        squeeze_back = False
    else:
        raise ValueError("Input must be [T] or [B,T]")

    B, T = f0_old.shape
    device = f0_old.device
    f0_old = f0_old.to(dtype=torch.float32)
    f0_new = f0_new.to(dtype=torch.float32)

    # 全局变化 mask
    finite_mask = torch.isfinite(f0_old) & torch.isfinite(f0_new)
    changed_mask = (torch.abs(f0_new - f0_old) > atol_hz) & finite_mask

    # 统计整体
    total_changed = int(changed_mask.sum().item())
    has_change = total_changed > 0
    overall = {
        'batch_size': B,
        'frames': T,
        'changed_frames': total_changed,
        'changed_ratio': float(total_changed / (B * T))
    }

    per_sample = []

    for i in range(B):
        x0 = f0_old[i]
        x1 = f0_new[i]
        cm = changed_mask[i]

        # 基本统计（Hz）
        n_changed = int(cm.sum().item())
        abs_diff = torch.abs(x1 - x0)
        abs_hz_mean = float(abs_diff[cm].mean().item()) if n_changed > 0 else 0.0
        abs_hz_max  = float(abs_diff[cm].max().item())  if n_changed > 0 else 0.0

        # 计算 Δcents（仅 voiced）
        delta_cents, voiced_mask = _safe_delta_cents(x0, x1, ref_hz_for_cents, silence_threshold)
        voiced_changed = cm & voiced_mask
        n_vc = int(voiced_changed.sum().item())
        abs_cents_mean = float(delta_cents[voiced_changed].abs().mean().item()) if n_vc > 0 else 0.0
        abs_cents_max  = float(delta_cents[voiced_changed].abs().max().item())  if n_vc > 0 else 0.0

        # 找连续变化片段（基于 changed_mask）
        segs = _contiguous_true_runs(cm)
        seg_info = []
        for (s, e) in segs:
            L = e - s + 1
            seg_old = x0[s:e+1]
            seg_new = x1[s:e+1]
            seg_abs_hz = torch.abs(seg_new - seg_old)
            seg_delta_c = delta_cents[s:e+1]
            seg_voiced  = voiced_mask[s:e+1]
            vc = seg_voiced.sum().item()

            # 片段统计
            d_hz_mean = float(seg_abs_hz.mean().item())
            d_hz_max  = float(seg_abs_hz.max().item())
            if vc >= 2:
                d_c_mean = float(seg_delta_c[seg_voiced].mean().item())
                d_c_std  = float(seg_delta_c[seg_voiced].std(unbiased=False).item())
            else:
                d_c_mean, d_c_std = 0.0, 0.0

            # 片段类型判别（粗略）：
            seg_type = 'unvoiced-or-mixed'
            glide_slope_pf = 0.0
            glide_slope_ps = 0.0
            glide_r2 = 0.0
            if vc >= 3:
                # 拟合 Δcents ~ a * n + b
                y = seg_delta_c[seg_voiced]
                n = torch.arange(y.numel(), dtype=torch.float32, device=device)
                x_mean = n.mean()
                y_mean = y.mean()
                var_x = torch.sum((n - x_mean) ** 2)
                if var_x > 0:
                    cov_xy = torch.sum((n - x_mean) * (y - y_mean))
                    a = cov_xy / var_x  # slope (cents / frame)
                    b = y_mean - a * x_mean
                    y_hat = a * n + b
                    ss_res = torch.sum((y - y_hat) ** 2)
                    ss_tot = torch.sum((y - y_mean) ** 2)
                    r2 = float(1.0 - (ss_res / ss_tot + 1e-12))
                    glide_slope_pf = float(a.item())
                    glide_slope_ps = float(a.item() * (sample_rate_hz if sample_rate_hz else 1.0))
                    glide_r2 = r2

                    # 分类规则：
                    #   jump-like: Δcent 在片段内近似常数（std 很小）
                    #   glide-like: Δcent 近似线性（R^2 高）
                    #   否则：jitter-like
                    if d_c_std <= jump_std_max_cents:
                        seg_type = 'jump-like'
                    elif r2 >= glide_r2_min:
                        seg_type = 'glide-like'
                    else:
                        seg_type = 'jitter-like'
                else:
                    # x 方差为 0（极短）时，退化判断
                    seg_type = 'jump-like' if d_c_std <= jump_std_max_cents else 'jitter-like'

            # 时间戳
            if sample_rate_hz:
                t0 = s / sample_rate_hz
                t1 = e / sample_rate_hz
            else:
                t0 = None
                t1 = None

            seg_info.append({
                'start': s, 'end': e, 'len': L,
                'start_time': t0, 'end_time': t1,
                'delta_hz_mean': d_hz_mean,
                'delta_hz_max': d_hz_max,
                'delta_cents_mean_voiced': d_c_mean,
                'delta_cents_std_voiced': d_c_std,
                'glide_slope_cents_per_frame': glide_slope_pf,
                'glide_slope_cents_per_sec': glide_slope_ps,
                'glide_r2': glide_r2,
                'type': seg_type,
            })

        per_sample.append({
            'idx': i if not squeeze_back else 0,
            'changed_frames': n_changed,
            'ratio': float(n_changed / T),
            'abs_hz_mean': abs_hz_mean,
            'abs_hz_max':  abs_hz_max,
            'abs_cents_mean_voiced': abs_cents_mean,
            'abs_cents_max_voiced':  abs_cents_max,
            'segments': seg_info,
        })

    return {
        'has_change': has_change,
        'overall': overall,
        'per_sample': per_sample
    }
