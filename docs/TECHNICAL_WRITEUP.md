# Technical Writeup — Decoding Hand Movement from ECoG

*A team project for BE 5210 (Brain–Computer Interfaces), University of Pennsylvania, Spring 2026, by Nandagopal Vidhu, Qingyuan Shi, and Samir Patki. This document re-tells the methods and findings in my own words; the work was a joint effort.*

## 1. Problem

The goal is to decode continuous five-finger flexion from electrocorticography (ECoG). For each of three human subjects we are given a 1 kHz ECoG recording (62, 48, and 64 channels respectively) paired with a 5-dimensional dataglove signal measuring finger position over time. The task is a continuous regression — reconstruct the glove trajectory from the brain signal — and the competition metric is the mean Pearson correlation across fingers 1, 2, 3, and 5 (the ring finger, 4, is excluded from scoring).

The final pipeline reached a leaderboard correlation of **_r_ ≈ 0.6808** using a per-subject Temporal Convolutional Network (TCN) on hand-crafted spectral features.

## 2. Data and splits

The dataset is organized as 3×1 cell arrays with subject-specific channel dimensions. We used the provided training data with an **80/20 chronological split** (not random — the signal is a time series, so a random split would leak future context into the validation set). The same preprocessing, feature extraction, and prediction pipeline was used to generate predictions on the leaderboard and hidden test sets.

## 3. Preprocessing

Raw ECoG is noisy and shares global artifacts across electrodes, so we clean it before extracting features:

1. **Common-average reference (CAR):** subtract the across-channel mean at each time point, removing artifacts shared by all electrodes.
2. **Notch filtering:** second-order IIR notch filters at 60 Hz and its harmonics (120 Hz, 180 Hz), quality factor Q = 30, to suppress power-line interference.
3. **Bandpass:** a 4th-order Butterworth bandpass, 0.15–200 Hz.

All filtering is applied per channel with zero-phase forward–backward filtering (`scipy.signal.filtfilt`) to avoid phase distortion.

## 4. Feature design

We use a sliding-window representation: 100 ms windows with 50 ms overlap, giving an effective feature rate of 20 Hz. For each window and each channel we compute **10 features**:

- **Time domain (3):** mean voltage, variance, and average line-length (mean absolute first difference).
- **Band RMS power (6):** root-mean-square amplitude in six bands — 5–15, 20–25, 30–50, 70–100, 100–150, and 150–200 Hz — capturing activity at different temporal scales.
- **High-gamma envelope (1):** mean of the Hilbert-transform amplitude envelope of the 70–200 Hz band. High-gamma power is a well-established correlate of local cortical activation and is the single most informative movement feature.

Concatenating across channels gives feature vectors of dimension 620 / 480 / 640 for the three subjects. The glove targets are down-sampled onto the same 20 Hz window grid by averaging within each window, so features and targets are aligned. Features are standardized (zero mean, unit variance) using statistics fit on the training split only.

## 5. Model: per-subject TCN

We train a **separate TCN for each subject**, because the channel counts differ (62/48/64) and the neural response patterns are subject-specific — any shared model loses to three specialized ones.

Each model takes a sequence of `seq_len ≈ 41` consecutive feature windows (~2 s of context) and predicts the glove value at the final window. The architecture:

- **5 stacked TemporalBlocks** with dilations 1, 2, 4, 8, 16, kernel size 3, hidden dimension 256.
- Each TemporalBlock is two dilated causal convolutions with ReLU, dropout (0.20–0.25), and a residual connection (a 1×1 convolution matches dimensions when input/output channels differ).
- **Causal padding + `Chomp1d` truncation** guarantee the model never sees future samples — essential for an honest BCI decoder.
- A fully connected head (256 → 128 → 5, ReLU) produces the five finger predictions from the last time step.

The five dilated levels cover a receptive field of ~125 feature windows (~6 s) at fixed depth — long temporal context bought through dilation rather than width, which matters because the per-subject training sets are small.

## 6. Training

Models are trained per subject with **Adam** (learning rates 7×10⁻⁴ to 1×10⁻³, weight decay 1×10⁻⁴ to 3×10⁻⁴), batch size 128, for ~30–45 epochs, with subject-specific configurations. The objective is **MSE** between predicted and true flexion across all five fingers. During training we monitor the competition metric (mean correlation on fingers 1, 2, 3, 5) on the validation split and keep the **best-validation checkpoint** per subject. Internal validation correlations were 0.50 / 0.57 / 0.71 for subjects 1 / 2 / 3.

## 7. Postprocessing

The TCN predicts on the 20 Hz window grid; we map back to the original 1 kHz sample grid:

1. **Interpolation:** use each window's center index as a time anchor and linearly interpolate the window-level predictions to full resolution. Boundary values outside the valid prediction range are held at the nearest valid prediction (no extrapolation artifacts).
2. **Smoothing:** a Savitzky–Golay filter (window 51, polynomial order 3) followed by a length-20 moving-average boxcar per finger.
3. Pack the three subjects' predictions into the required 3×1 cell array `predicted_dg`, each an N×5 matrix aligned to the corresponding ECoG recording.

## 8. Approaches we explored and abandoned

The final pipeline is the survivor of several that we tried and discarded for concrete reasons (numbers are the best validation or leaderboard correlations on the scored fingers):

- **Linear & ridge regression on hand-crafted band features (_r_ ≈ 0.40–0.42).** Even with the high-gamma envelope, finer sub-bands, and a per-finger ridge sweep, both decoders plateaued. An inner product of features cannot capture non-linear or delayed coupling between frequency bands and movement.
- **Heavy temporal smoothing with cubic-spline up-sampling (_r_ ≈ 0.36).** Aggressive smoothing removed genuine fast transitions, especially thumb taps. Never submitted.
- **RNN + TCN hybrid stacks.** Adding LSTM/GRU layers on top of the TCN added parameters and overfit without improving accuracy — the dilated stack already covers the relevant motor-planning window.

### The activity-gated gradient-boosting decoder (_r_ ≈ 0.483, leaderboard)

This was our strongest non-neural-network pipeline, and worth describing in full because of what it taught us. It is a **two-stage gated decoder** that factorizes the prediction into *whether* a finger is moving and *by how much*:

- **Activity detector.** We pseudo-label each window as "active" when the glove envelope exceeds its 75th percentile, then train a class-balanced logistic regression on the features to output a per-window probability of movement.
- **Amplitude model.** For each subject and finger, a `HistGradientBoostingRegressor` is trained **only on active windows** (config chosen by a small sweep on the 80/20 split) to predict flexion magnitude.
- **Combination.** The prediction is `detector_probability × amplitude`, both lightly smoothed, then cubic-spline up-sampled to 1 kHz. The gate keeps the output near rest during quiet periods and only commits to large flexion when the detector fires.

The decoder was **notably accurate on the thumb**. For subject 1, a single high-gamma channel — **channel 42, AUROC ≈ 0.71** for thumb-active vs. rest — is so discriminative that a gate built on it alone produces a sharp, well-timed thumb prediction. This is the concrete finding the experiment contributed back to the main effort: *gating amplitude by movement onset helps the thumb specifically*, where one electrode carries a clean motor signal.

We abandoned it as the headline because it was **brittle**: the activity gate flickered on the hardest subject (subject 2) and needed several per-subject fall-backs, a sign of overfitting to a heuristic rather than a robust method. That fragility — a hand-tuned pipeline that doesn't generalize across subjects — is exactly what pushed us toward the per-subject TCN, which learns the temporal structure end-to-end and reached _r_ ≈ 0.68.

## 9. Why the ring finger behaves like a mix of the middle and little fingers

A recurring observation: the predicted ring-finger (digit 4) trace looks like a noisy blend of the middle (3) and little (5) fingers. Three converging reasons:

- **Mechanical coupling.** The flexor digitorum profundus muscle belly is fused across digits 2–5, and the *juncturae tendinum* tether the ring tendon to its neighbours, so a command for digit 4 drags 3 and 5 with it.
- **Behavioural.** The ring finger has the lowest individuation index of all five digits (Hager-Ross & Schieber, 2000): a clean ring-only movement is the hardest single-finger motion people produce, so the training data essentially never contains a ring-only event.
- **Neural.** Single-finger fields in primary motor cortex are graded and heavily overlapping, and ECoG strips average over millimetre-scale patches — so the neural code for finger 4 looks like a mixture of the codes for 3 and 5. Under squared-error loss the optimal estimator of that mixture is exactly the weighted average the TCN learns from the joint 5-finger target, which is why this coupling is absorbed for free without any explicit constraint. (Conveniently, finger 4 is also excluded from the competition metric.)

## 10. Conclusions and next steps

The clearest lesson was that **architecture choices mattered more than features**. A linear decoder cleared the first checkpoint but plateaued across bands and time; the gradient-boosting hybrid closed half of the remaining gap but was less effective on the hardest subject; the TCN did best because its dilated receptive field gave several seconds of strictly-causal context essentially for free in depth.

The main challenges were the differing channel counts across subjects (forcing per-subject models) and subject 2's lower SNR (validation correlations on some fingers stayed near zero across multiple feature sets), plus the smoothing-versus-responsiveness trade-off (aggressive smoothing helps the slow glove components but hurts the thumb).

Natural next steps: a learned feature extractor (a small Conv1D front-end on the filtered raw ECoG, replacing the hand-crafted band features), a calibrated ensemble of the TCN with the ridge and gradient-boosting decoders, and a substantially finer per-subject hyperparameter sweep — particularly for subject 2.

## References

- J. Kubanek et al., "Decoding flexion of individual fingers using electrocorticographic signals in humans," *Journal of Neural Engineering*, 2009.
- C. E. Hager-Ross & M. H. Schieber, "Quantifying the independence of human finger movements," *Journal of Neuroscience*, 2000.
- BCI Competition IV — finger-flexion (ECoG) dataset.
