# Dataset

**The data is not included in this repository.** It is course-provided material for BE 5210 (derived from the BCI Competition IV finger-flexion / ECoG paradigm) and is not ours to redistribute. This page documents what the data is and how the notebooks expect it to be laid out, so the work is reproducible if you have access to the dataset.

## What the data is

Paired electrocorticography (ECoG) and dataglove recordings from **three human epilepsy patients** who had subdural electrode grids implanted for clinical monitoring:

- **ECoG:** continuous voltage from cortical-surface electrodes, sampled at **1 kHz**. The three subjects have **62, 48, and 64 channels** respectively.
- **Dataglove:** continuous flexion of all **five fingers**, recorded simultaneously and aligned to the ECoG. This is the prediction target.

The decoding paradigm follows Kubanek et al. (2009), "Decoding flexion of individual fingers using electrocorticographic signals in humans," and the BCI Competition IV finger-flexion dataset.

## Expected files and layout

The notebooks read MATLAB `.mat` files via `scipy.io.loadmat`. Place them in this `data/` directory (or point the `DATA_DIR` constant at the top of each notebook to wherever they live):

```
data/
├── raw_training_data.mat     # training set: train_ecog, train_dg
└── leaderboard_data.mat      # held-out leaderboard ECoG: leaderboard_ecog
```

Structure of each variable (MATLAB cell arrays, one entry per subject):

| Variable | Shape | Contents |
|---|---|---|
| `train_ecog` | 3×1 cell | each cell is `(n_samples, n_channels)` ECoG, e.g. `(300000, 62)` |
| `train_dg`   | 3×1 cell | each cell is `(n_samples, 5)` finger flexion |
| `leaderboard_ecog` | 3×1 cell | each cell is `(n_samples, n_channels)` ECoG to predict on |

The training recordings are 300,000 samples (~5 min at 1 kHz) per subject; the leaderboard recordings are 147,500 samples.

## Outputs

Predictions are packed into a 3×1 cell array `predicted_dg`, each entry an `(n_samples, 5)` matrix aligned to the corresponding leaderboard ECoG, and saved to a `.mat` file for scoring.
