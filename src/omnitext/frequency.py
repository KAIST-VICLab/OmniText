"""Frequency-domain low-pass filters used for attention smoothing.

Extracted verbatim from the original OmniText notebooks.
"""

import math
import torch.fft as fft

def freq_mix_2d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    noise_freq = fft.fftn(noise, dim=(-2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-2, -1)).real

    return x_mixed

def butterworth_low_pass_filter(shape, device, n=4, d_s=0.25):
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
    """
    H, W = shape[-2], shape[-1]
    mask = torch.zeros(shape).to(device=device)
    if d_s==0:
        return mask
    
    for h in range(H):
        for w in range(W):
            d_square = ((2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2)
            value = 1 / (1 + (d_square / d_s**2) ** n)
            mask[..., h, w] = value
    return mask
    
# ---------------------------------------------------------------------
#  Utility – produce a (H, W) distance grid normalised to [0, 0.5]
# ---------------------------------------------------------------------
def _radius_grid(H: int, W: int, device=None):
    """Return radial distance ρ[i,j] ∈ [0, 0.5] from the spectrum centre."""
    ys = torch.arange(-H//2, H//2, device=device).float() / H   # [-0.5, 0.5)
    xs = torch.arange(-W//2, W//2, device=device).float() / W
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    rho = (xx**2 + yy**2).sqrt()               # Euclidean radius
    return rho                                  # shape (H, W)

# ---------------------------------------------------------------------
#  1. IDEAL (“brick‑wall”) LPF
# ---------------------------------------------------------------------
def ideal_lowpass(H, W, cutoff=0.15, device=None):
    """
    cutoff ∈ (0, 0.5) is the normalised pass‑band radius.
    """
    rho = _radius_grid(H, W, device)
    return (rho <= cutoff).float()              # hard 0/1 mask

# ---------------------------------------------------------------------
#  2. GAUSSIAN LPF
# ---------------------------------------------------------------------
def gaussian_lowpass(H, W, sigma=0.07, device=None):
    """
    sigma : standard deviation of the Gaussian in normalised radius units.
    """
    rho = _radius_grid(H, W, device)
    return torch.exp(-(rho ** 2) / (2 * sigma ** 2))

# ---------------------------------------------------------------------
#  3. BUTTERWORTH LPF
# ---------------------------------------------------------------------
def butterworth_lowpass(H, W, cutoff=0.15, order=4, device=None):
    """
    cutoff : −3 dB point (normalised radius)
    order  : filter order; higher = steeper transition.
    """
    rho = _radius_grid(H, W, device)
    # Avoid divide‑by‑zero at the origin
    eps  = 1e-9
    return 1 / (1 + (rho / (cutoff + eps)) ** (2 * order))

# ---------------------------------------------------------------------
#  4. RAISED‑COSINE / HANN LPF (“Feathered” ideal)
# ---------------------------------------------------------------------
def raised_cosine_lowpass(H, W, cutoff=0.15, width=0.05, device=None):
    """
    cutoff : start of the transition band.
    width  : width of the cosine roll‑off (0 < width ≤ cutoff).
    """
    rho   = _radius_grid(H, W, device)
    lpf   = torch.zeros_like(rho)
    passb = rho <= cutoff                       # full pass
    trans = (rho > cutoff) & (rho <= cutoff + width)
    # Cosine taper in the transition region
    lpf[passb]  = 1.0
    lpf[trans]  = 0.5 * (1 + torch.cos(math.pi *
                         (rho[trans] - cutoff) / width))
    # else remains 0 (stop‑band)
    return lpf
