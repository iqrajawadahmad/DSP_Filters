import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, filtfilt


FS = 1000
DURATION = 2
NUM_TAPS = 201


def moving_average(signal, window_size=11):
    """Smooth a signal using a normalized FIR moving-average filter."""
    coefficients = np.ones(window_size) / window_size
    return filtfilt(coefficients, [1.0], signal)


def make_filters(fs):
    """Create common FIR filters and their design parameters."""
    return {
        "Low-pass (100 Hz)": firwin(NUM_TAPS, 100, fs=fs),
        "High-pass (50 Hz)": firwin(NUM_TAPS, 50, pass_zero=False, fs=fs),
        "Band-pass (40-120 Hz)": firwin(
            NUM_TAPS, [40, 120], pass_zero=False, fs=fs
        ),
        "Band-stop (70-90 Hz)": firwin(
            NUM_TAPS, [70, 90], pass_zero="bandstop", fs=fs
        ),
    }


def one_sided_amplitude(signal, fs):
    """Return the correctly scaled one-sided amplitude spectrum."""
    sample_count = len(signal)
    spectrum = np.abs(np.fft.rfft(signal)) / sample_count
    if sample_count % 2 == 0:
        spectrum[1:-1] *= 2
    else:
        spectrum[1:] *= 2
    frequencies = np.fft.rfftfreq(sample_count, d=1 / fs)
    return frequencies, spectrum


def main():
    np.random.seed(42)
    time = np.arange(0, DURATION, 1 / FS)

    clean = (
        np.sin(2 * np.pi * 20 * time)
        + 0.7 * np.sin(2 * np.pi * 80 * time)
        + 0.4 * np.sin(2 * np.pi * 200 * time)
    )
    noisy = clean + 0.8 * np.random.randn(len(time))

    filter_coefficients = make_filters(FS)
    outputs = {
        name: filtfilt(coefficients, [1.0], noisy)
        for name, coefficients in filter_coefficients.items()
    }
    outputs["Moving average (11 samples)"] = moving_average(noisy, 11)

    # Time-domain comparison
    time_signals = [("Clean reference", clean), ("Noisy input", noisy)]
    time_signals.extend(outputs.items())
    fig, axes = plt.subplots(
        len(time_signals), 1, figsize=(12, 14), sharex=True
    )
    for axis, (label, signal) in zip(axes, time_signals):
        axis.plot(time, signal, linewidth=0.8)
        axis.set_title(label)
        axis.set_ylabel("Amplitude")
        axis.grid(True)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("FIR Filter Time-Domain Comparison")
    fig.tight_layout()
    fig.savefig("signal_comparison.png", dpi=150)
    plt.close(fig)

    # Frequency-domain comparison
    fig, axis = plt.subplots(figsize=(12, 6))
    for label, signal in [
        ("Clean reference", clean),
        ("Noisy input", noisy),
        ("Low-pass output", outputs["Low-pass (100 Hz)"]),
        ("Band-pass output", outputs["Band-pass (40-120 Hz)"]),
    ]:
        frequencies, amplitude = one_sided_amplitude(signal, FS)
        axis.plot(frequencies, amplitude, label=label)
    axis.set_title("One-Sided Amplitude Spectrum")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Amplitude")
    axis.set_xlim(0, 300)
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig("fft_analysis.png", dpi=150)
    plt.close(fig)

    # Designed FIR filter frequency responses
    fig, axis = plt.subplots(figsize=(12, 6))
    for label, coefficients in filter_coefficients.items():
        frequencies, response = freqz(coefficients, worN=2048, fs=FS)
        axis.plot(frequencies, 20 * np.log10(np.maximum(np.abs(response), 1e-6)),
                  label=label)
    axis.set_title("Designed FIR Filter Frequency Responses")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Magnitude (dB)")
    axis.set_xlim(0, FS / 2)
    axis.set_ylim(-80, 5)
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig("filter_responses.png", dpi=150)
    plt.close(fig)

    # Moving-average window comparison, shown over a short interval for clarity
    fig, axis = plt.subplots(figsize=(12, 5))
    visible_samples = int(0.25 * FS)
    axis.plot(
        time[:visible_samples], clean[:visible_samples],
        label="Clean reference", linewidth=2
    )
    for window_size in [3, 11, 31]:
        smoothed = moving_average(noisy, window_size)
        axis.plot(
            time[:visible_samples], smoothed[:visible_samples],
            label=f"Window = {window_size}"
        )
    axis.set_title("Moving-Average Filter: Window-Size Comparison")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig("moving_average.png", dpi=150)
    plt.close(fig)

    print("DSP filtering demo completed.")
    print("Generated: signal_comparison.png")
    print("Generated: fft_analysis.png")
    print("Generated: filter_responses.png")
    print("Generated: moving_average.png")
    print("Filters: low-pass, high-pass, band-pass, band-stop, moving average")


if __name__ == "__main__":
    main()
