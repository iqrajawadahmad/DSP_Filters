# Basic DSP Signal Filtering

A small Python project demonstrating common digital signal-processing filters
on a noisy, multi-tone signal.

## Filters demonstrated

- **Low-pass:** keeps frequencies below 100 Hz.
- **High-pass:** keeps frequencies above 50 Hz.
- **Band-pass:** keeps frequencies between 40 and 120 Hz.
- **Band-stop:** attenuates frequencies between 70 and 90 Hz.
- **Moving average:** smooths the signal with different window sizes.

The four designed filters use the FIR window method (`scipy.signal.firwin`).
Filtering uses forward-backward filtering, so the plotted results have no
phase delay; this is suitable for an offline demonstration.

## Run

Install the dependencies and run the script:

```bash
python -m pip install -r requirements.txt
python app.py
```

The script saves these plots in the project folder:

- `signal_comparison.png` — clean, noisy, and filtered signals in the time domain.
- `fft_analysis.png` — one-sided amplitude spectra before and after filtering.
- `filter_responses.png` — designed FIR filter frequency responses.
- `moving_average.png` — smoothing comparison for three window sizes.

The signal contains 20 Hz, 80 Hz, and 200 Hz components. Filter outputs should
be interpreted according to each filter's passband: a filter that intentionally
removes one of these components is not expected to reproduce the entire clean
reference signal.
