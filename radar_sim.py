"""
FMCW mmWave radar simulation (60 GHz, similar to TI IWR6843).

Pipeline:
  1. Simulate the IF (beat) signal for several targets (walking person, static wall)
  2. Range FFT  -> distance
  3. Static clutter removal (subtract mean over chirps) -> removes wall/furniture
  4. Doppler FFT -> velocity
  5. CA-CFAR detection on the range-Doppler map
  6. Plot range profile + range-Doppler map with detections

Run:  python radar_sim.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

C = 3e8  # speed of light (m/s)

# ---------------- Radar parameters ----------------
F0 = 60e9          # start frequency (Hz)
BANDWIDTH = 4e9    # sweep bandwidth (Hz) -> range resolution = c / 2B = 3.75 cm
T_CHIRP = 40e-6    # chirp ramp duration (s)
T_REP = 200e-6     # chirp repetition interval incl. idle time (s)
N_SAMPLES = 256    # ADC samples per chirp
N_CHIRPS = 128     # chirps per frame
FS = N_SAMPLES / T_CHIRP  # ADC sample rate
SLOPE = BANDWIDTH / T_CHIRP
WAVELENGTH = C / F0

RANGE_RES = C / (2 * BANDWIDTH)
MAX_RANGE = FS * C / (2 * SLOPE)          # complex (IQ) sampling
VEL_RES = WAVELENGTH / (2 * N_CHIRPS * T_REP)
MAX_VEL = WAVELENGTH / (4 * T_REP)


def simulate_frame(targets, noise_std=0.05, seed=0):
    """Return complex IF data of shape (N_CHIRPS, N_SAMPLES).

    targets: list of (range_m, velocity_m_s, amplitude)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / FS
    data = np.zeros((N_CHIRPS, N_SAMPLES), dtype=complex)
    for r0, v, amp in targets:
        for k in range(N_CHIRPS):
            r = r0 + v * k * T_REP
            f_beat = 2 * SLOPE * r / C
            phase = 4 * np.pi * r / WAVELENGTH
            data[k] += amp * np.exp(1j * (2 * np.pi * f_beat * t + phase))
    noise = rng.normal(0, noise_std, data.shape) + 1j * rng.normal(0, noise_std, data.shape)
    return data + noise


def range_fft(data):
    win = np.hanning(N_SAMPLES)
    return np.fft.fft(data * win, axis=1)


def remove_static_clutter(rfft):
    return rfft - rfft.mean(axis=0, keepdims=True)


def doppler_fft(rfft):
    win = np.hanning(N_CHIRPS)[:, None]
    return np.fft.fftshift(np.fft.fft(rfft * win, axis=0), axes=0)


def ca_cfar_2d(power, guard=2, train=8, scale_db=12.0):
    """Cell-averaging CFAR on a 2D power map (linear). Returns boolean mask."""
    outer = 2 * (guard + train) + 1
    inner = 2 * guard + 1
    sum_outer = uniform_filter(power, outer, mode="wrap") * outer**2
    sum_inner = uniform_filter(power, inner, mode="wrap") * inner**2
    noise = (sum_outer - sum_inner) / (outer**2 - inner**2)
    return power > noise * 10 ** (scale_db / 10)


def axes():
    ranges = np.arange(N_SAMPLES) * RANGE_RES
    vels = (np.arange(N_CHIRPS) - N_CHIRPS // 2) * VEL_RES
    return ranges, vels


def detect(targets, **kw):
    """Full pipeline. Returns list of (range_m, velocity_m_s) detections and the RD map."""
    data = simulate_frame(targets, **kw)
    rfft = remove_static_clutter(range_fft(data))
    rd = doppler_fft(rfft)
    power = np.abs(rd) ** 2
    mask = ca_cfar_2d(power)
    ranges, vels = axes()
    # keep only the strongest cell per connected peak (simple local-max check)
    dets = []
    for d, r in zip(*np.nonzero(mask)):
        win = power[max(d - 2, 0):d + 3, max(r - 2, 0):r + 3]
        if power[d, r] == win.max():
            dets.append((ranges[r], vels[d]))
    return dets, rd


def main():
    targets = [
        (2.0, 0.8, 1.0),    # person walking away at 0.8 m/s, 2 m
        (3.5, -0.5, 0.6),   # second person approaching
        (5.0, 0.0, 3.0),    # static wall (strong) -> removed by clutter filter
    ]
    print(f"Range resolution: {RANGE_RES*100:.2f} cm | max range: {MAX_RANGE:.1f} m")
    print(f"Velocity resolution: {VEL_RES:.3f} m/s | max velocity: +/-{MAX_VEL:.2f} m/s")

    data = simulate_frame(targets)
    raw_profile = np.abs(range_fft(data)).mean(axis=0)
    dets, rd = detect(targets)
    ranges, vels = axes()

    for r, v in dets:
        print(f"Detected target: range {r:.2f} m, velocity {v:+.2f} m/s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ranges, 20 * np.log10(raw_profile))
    ax1.set(xlabel="Range (m)", ylabel="Magnitude (dB)", title="Range profile (before clutter removal)", xlim=(0, 8))
    img = ax2.imshow(20 * np.log10(np.abs(rd) + 1e-9), aspect="auto", origin="lower",
                     extent=[ranges[0], ranges[-1], vels[0], vels[-1]], cmap="viridis")
    for r, v in dets:
        ax2.plot(r, v, "rx", ms=10, mew=2)
    ax2.set(xlabel="Range (m)", ylabel="Velocity (m/s)", title="Range-Doppler map + CFAR detections", xlim=(0, 8))
    fig.colorbar(img, ax=ax2, label="dB")
    fig.tight_layout()
    fig.savefig("range_doppler.png", dpi=150)
    print("Saved range_doppler.png")


if __name__ == "__main__":
    main()
