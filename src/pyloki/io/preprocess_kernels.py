"""Numba-compiled numerical kernels for timeseries preprocessing."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

K_MAD_SCALE = 1.482602218505602


@njit(cache=True)
def _min_valid(min_good: float, length: int, block: int) -> int:
    need = math.ceil(min_good * length)
    floor_val = min(3, block)
    return max(floor_val, need)


@njit(cache=True)
def fill_holes(values: np.ndarray, have: np.ndarray) -> bool:
    """Fill entries with ``have == False`` by linear interpolation in place.

    Returns ``False`` if no entry has ``have == True``.
    """
    n = values.size
    prev = -1
    for i in range(n):
        if not have[i]:
            continue
        if prev == -1:
            val = values[i]
            for j in range(i):
                values[j] = val
        elif i > prev + 1:
            a = values[prev]
            d = values[i] - a
            span = float(i - prev)
            for j in range(prev + 1, i):
                values[j] = a + d * (j - prev) / span
        prev = i
    if prev == -1:
        return False
    last_val = values[prev]
    for j in range(prev + 1, n):
        values[j] = last_val
    return True


@njit(cache=True)
def reflected_running_mean(values: np.ndarray, k: int) -> np.ndarray:
    """Centred running mean over ``k`` entries with point-reflected edges."""
    n = values.size
    half = k // 2
    if half == 0 or n < 2:
        return values.copy()
    ext_len = n + 2 * half
    prefix = np.empty(ext_len + 1, dtype=np.float64)
    prefix[0] = 0.0
    for idx in range(ext_len):
        j = idx - half
        if j < 0:
            m = min(n - 1, -j)
            val = 2.0 * values[0] - values[m]
        elif j >= n:
            m = 2 * (n - 1) - j
            m = max(m, 0)
            val = 2.0 * values[n - 1] - values[m]
        else:
            val = values[j]
        prefix[idx + 1] = prefix[idx] + val

    out = np.empty(n, dtype=np.float64)
    width = float(2 * half + 1)
    for i in range(n):
        out[i] = (prefix[i + 2 * half + 1] - prefix[i]) / width
    return out


@njit(cache=True)
def masked_running_median(
    values: np.ndarray,
    usable: np.ndarray,
    k: int,
) -> tuple[np.ndarray, bool]:
    """Centred running median of usable entries with point-reflected edges."""
    n = values.size
    filled = values.copy()
    if not fill_holes(filled, usable):
        return np.zeros(n, dtype=np.float64), False

    half = k // 2
    if half == 0 or n < 2:
        return filled, True

    first_ok = 0
    for i in range(n):
        if usable[i]:
            first_ok = i
            break
    last_ok = n - 1
    for i in range(n - 1, -1, -1):
        if usable[i]:
            last_ok = i
            break

    width = 2 * half + 1
    buf = np.empty(width, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    have = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        cnt = 0
        for offset in range(-half, half + 1):
            j = i + offset
            if j < 0:
                m = first_ok - j
                m = min(m, n - 1)
                buf[cnt] = 2.0 * filled[first_ok] - filled[m]
                cnt += 1
            elif j >= n:
                m = last_ok - (j - n + 1)
                m = max(m, 0)
                buf[cnt] = 2.0 * filled[last_ok] - filled[m]
                cnt += 1
            elif usable[j]:
                buf[cnt] = values[j]
                cnt += 1
        if cnt > 0:
            out[i] = np.median(buf[:cnt])
            have[i] = True

    ok = fill_holes(out, have)
    return out, ok


@njit(cache=True)
def masked_running_mean(
    values: np.ndarray,
    usable: np.ndarray,
    k: int,
) -> tuple[np.ndarray, bool]:
    """Centred running mean of usable entries with truncated edges."""
    n = values.size
    half = k // 2
    csum = np.empty(n + 1, dtype=np.float64)
    ccnt = np.empty(n + 1, dtype=np.int64)
    csum[0] = 0.0
    ccnt[0] = 0
    for i in range(n):
        if usable[i]:
            csum[i + 1] = csum[i] + values[i]
            ccnt[i + 1] = ccnt[i] + 1
        else:
            csum[i + 1] = csum[i]
            ccnt[i + 1] = ccnt[i]

    out = np.zeros(n, dtype=np.float64)
    have = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        lo = i - half if i >= half else 0
        hi = min(n, i + half + 1)
        c = ccnt[hi] - ccnt[lo]
        if c > 0:
            out[i] = (csum[hi] - csum[lo]) / float(c)
            have[i] = True

    ok = fill_holes(out, have)
    return out, ok


@njit(cache=True)
def interpolate_blocks(n: int, block: int, values: np.ndarray) -> np.ndarray:
    """Linearly interpolate block-centred values to all ``n`` samples."""
    nblocks = values.size
    out = np.empty(n, dtype=np.float64)
    if nblocks == 1:
        val = values[0]
        for i in range(n):
            out[i] = val
        return out

    centers = np.empty(nblocks, dtype=np.float64)
    for b in range(nblocks):
        b0 = b * block
        b1 = min(n, b0 + block)
        centers[b] = b0 + 0.5 * (b1 - b0 - 1)

    b = 0
    for i in range(n):
        fi = float(i)
        if fi <= centers[0]:
            out[i] = values[0]
        elif fi >= centers[nblocks - 1]:
            out[i] = values[nblocks - 1]
        else:
            while b + 1 < nblocks - 1 and centers[b + 1] < fi:
                b += 1
            t = (fi - centers[b]) / (centers[b + 1] - centers[b])
            out[i] = values[b] + t * (values[b + 1] - values[b])
    return out


@njit(cache=True)
def compute_block_medians(
    x: np.ndarray,
    mask: np.ndarray,
    block: int,
    min_good: float,
    all_valid: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-block median, usability flag, and valid fraction."""
    n = x.size
    nblocks = (n + block - 1) // block
    med = np.zeros(nblocks, dtype=np.float64)
    usable = np.zeros(nblocks, dtype=np.bool_)
    good = np.zeros(nblocks, dtype=np.float64)
    buf = np.empty(block, dtype=np.float64)

    for b in range(nblocks):
        b0 = b * block
        b1 = min(n, b0 + block)
        length = b1 - b0
        if all_valid:
            good[b] = 1.0
            med[b] = np.median(x[b0:b1])
            usable[b] = True
        else:
            cnt = 0
            for i in range(b0, b1):
                if mask[i]:
                    buf[cnt] = x[i]
                    cnt += 1
            good[b] = cnt / float(length)
            if cnt >= _min_valid(min_good, length, block):
                med[b] = np.median(buf[:cnt])
                usable[b] = True
    return med, usable, good


@njit(cache=True)
def compute_block_variances(
    x: np.ndarray,
    mask: np.ndarray,
    mu_interp: np.ndarray,
    block: int,
    min_good: float,
    all_valid: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-block MAD-squared variance and usability flag."""
    n = x.size
    nblocks = (n + block - 1) // block
    var = np.zeros(nblocks, dtype=np.float64)
    usable = np.zeros(nblocks, dtype=np.bool_)
    res = np.empty(block, dtype=np.float64)
    dev = np.empty(block, dtype=np.float64)

    for b in range(nblocks):
        b0 = b * block
        b1 = min(n, b0 + block)
        length = b1 - b0
        cnt = 0
        if all_valid:
            for i in range(b0, b1):
                res[cnt] = x[i] - mu_interp[i]
                cnt += 1
        else:
            for i in range(b0, b1):
                if mask[i]:
                    res[cnt] = x[i] - mu_interp[i]
                    cnt += 1
            if cnt < _min_valid(min_good, length, block):
                continue
        if cnt < 2:
            continue
        med_r = np.median(res[:cnt])
        for i in range(cnt):
            dev[i] = abs(res[i] - med_r)
        sigma = K_MAD_SCALE * np.median(dev[:cnt])
        v = sigma * sigma
        if v > 0.0:
            var[b] = v
            usable[b] = True
    return var, usable


@njit(cache=True)
def flag_bad_blocks_scale(
    z: np.ndarray,
    mask: np.ndarray,
    scale: int,
    nsigma: float,
    min_good: float,
    clip_sigma: float,
) -> None:
    """Flag bad blocks at a single block ``scale`` and update ``mask`` in place."""
    n = z.size
    nblocks = (n + scale - 1) // scale
    bad = np.zeros(nblocks, dtype=np.bool_)
    stat_m = np.zeros(nblocks, dtype=np.float64)
    stat_v = np.zeros(nblocks, dtype=np.float64)
    n_ok = 0

    for b in range(nblocks):
        b0 = b * scale
        b1 = min(n, b0 + scale)
        length = b1 - b0
        cnt = 0
        s1 = 0.0
        s2 = 0.0
        for i in range(b0, b1):
            if mask[i]:
                zi = z[i]
                if clip_sigma <= 0.0 or abs(zi) <= clip_sigma:
                    cnt += 1
                    s1 += zi
                    s2 += zi * zi
        if cnt == 0 or cnt < min_good * length:
            bad[b] = True
            continue
        c = float(cnt)
        stat_m[b] = (s1 / c) * math.sqrt(c)
        wh = 2.0 / (9.0 * c)
        mean_sq = s2 / c
        cbrt_val = mean_sq ** (1.0 / 3.0) if mean_sq > 0.0 else 0.0
        stat_v[b] = (cbrt_val - (1.0 - wh)) / math.sqrt(wh)
        n_ok += 1

    if n_ok > 0:
        vm = np.empty(n_ok, dtype=np.float64)
        vv = np.empty(n_ok, dtype=np.float64)
        k = 0
        for b in range(nblocks):
            if not bad[b]:
                vm[k] = stat_m[b]
                vv[k] = stat_v[b]
                k += 1
        cm = np.median(vm)
        cv = np.median(vv)
        dm = np.empty(n_ok, dtype=np.float64)
        dv = np.empty(n_ok, dtype=np.float64)
        for i in range(n_ok):
            dm[i] = abs(vm[i] - cm)
            dv[i] = abs(vv[i] - cv)
        sm = K_MAD_SCALE * np.median(dm)
        sv = K_MAD_SCALE * np.median(dv)
        sm = max(sm, 1.0)
        sv = max(sv, 1.0)
        for b in range(nblocks):
            if not bad[b] and (
                abs(stat_m[b] - cm) > nsigma * sm or abs(stat_v[b] - cv) > nsigma * sv
            ):
                bad[b] = True

    for b in range(nblocks):
        if bad[b]:
            b0 = b * scale
            b1 = min(n, b0 + scale)
            for i in range(b0, b1):
                mask[i] = False


@njit(cache=True)
def longest_zero_run(arr: np.ndarray) -> int:
    """Return the length of the longest contiguous run of zeros in ``arr``."""
    best = 0
    run = 0
    for i in range(arr.size):
        if arr[i] == 0.0:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best
