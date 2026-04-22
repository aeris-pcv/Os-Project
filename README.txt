Deepfake Detection Model — Complete Walkthrough Guide

A step-by-step guide to running the OS-Optimized Deepfake Detection notebook from scratch
on Google Colab, starting with zero accounts and ending with a trained and evaluated model.

================================================================================
TABLE OF CONTENTS
================================================================================

1. Get a Kaggle Account
2. Generate Your Kaggle API Key
3. Store Your Credentials in Colab Secrets
4. Open the Notebook in Google Colab
5. Enable GPU Acceleration
6. Run the Notebook — Cell by Cell
7. Expected Output & Results
8. Troubleshooting
9. Project Architecture Reference

================================================================================
1. GET A KAGGLE ACCOUNT
================================================================================

If you already have a Kaggle account, skip to Step 2.

1. Go to https://www.kaggle.com
2. Click Register in the top-right corner
3. Sign up with Google, or enter your name, email, and a password
4. Verify your email address via the confirmation link Kaggle sends you
5. Complete your profile — you will be asked to choose a username

Note your username. It appears in the URL when you visit your profile page:
kaggle.com/<your-username>. You will need it in Step 3.

================================================================================
2. GENERATE YOUR KAGGLE API KEY
================================================================================

The notebook downloads the dataset automatically using the Kaggle CLI.
For this to work you need an API key.

1. Log in to https://www.kaggle.com
2. Click your profile picture (top-right) -> Settings
3. Scroll down to the API section
4. Click Create New Token

Kaggle will display your API key as a plain text string ONE TIME ONLY:

    a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4

Copy it immediately and save it somewhere safe (a password manager, a private
note). Once you close or navigate away from the page, Kaggle will not show it
again. If you lose it, return to Settings -> API -> Create New Token to generate
a fresh one — the old key will be invalidated automatically.

SECURITY: Never paste your API key directly into a notebook cell. That is
exactly why we store it in Colab Secrets in the next step.

================================================================================
3. STORE YOUR CREDENTIALS IN COLAB SECRETS
================================================================================

Google Colab has a built-in Secrets manager that keeps sensitive values out of
your notebook code entirely.

How to open Colab Secrets:
  1. Open your notebook in https://colab.research.google.com
  2. In the left sidebar, click the key icon (hover to see the label "Secrets")
  3. The Secrets panel will slide open

Add your Kaggle username:
  1. Click + Add new secret
  2. Name:  KAGGLE_USERNAME
  3. Value: your Kaggle username (e.g. johndoe)
  4. Toggle Notebook access -> ON

Add your Kaggle API key:
  1. Click + Add new secret again
  2. Name:  KAGGLE_KEY
  3. Value: paste the API key string you copied in Step 2
  4. Toggle Notebook access -> ON

Your Secrets panel should now show:

  Name               Notebook access
  KAGGLE_USERNAME    ON
  KAGGLE_KEY         ON

The names must be typed EXACTLY as shown. The notebook reads them with
userdata.get('KAGGLE_USERNAME') and userdata.get('KAGGLE_KEY') — any typo
will cause a SecretNotFoundError at runtime.

================================================================================
4. OPEN THE NOTEBOOK IN GOOGLE COLAB
================================================================================

1. Go to https://colab.research.google.com
2. Click File -> Upload notebook
3. Select OS_Supream_Ultimate_LAST_VERSION.ipynb from your computer
4. Colab will open it with all cells and structure intact

If the notebook is already saved in your Google Drive:
  1. Go to https://drive.google.com
  2. Navigate to the .ipynb file and double-click it — it opens directly in Colab

================================================================================
5. ENABLE GPU ACCELERATION
================================================================================

The notebook runs on CPU but is dramatically faster with a GPU.
Colab offers free GPU access.

1. In the top menu click Runtime -> Change runtime type
2. Under Hardware accelerator select T4 GPU
3. Click Save
4. Wait for Colab to reconnect — the green checkmark will reappear top-right

You will confirm the GPU is live when Cell 2 prints:

  CUDA available: True
  GPU: Tesla T4

If you see "CUDA available: False" after setting the GPU, go to
Runtime -> Restart session, then re-run from Cell 1.

================================================================================
6. RUN THE NOTEBOOK — CELL BY CELL
================================================================================

Run each cell in order with Shift + Enter.

------------------------------------------------------------------------
Cell 1 — Kaggle Setup & Dataset Download
------------------------------------------------------------------------
Reads your credentials from Colab Secrets, sets them as environment
variables, installs the Kaggle CLI, and downloads + unzips the dataset
to /content/.

Expected output:
  Downloading deepfake-and-real-images.zip to /content/
  100%|████████████████| 1.24G/1.24G [00:38<00:00, 35.1MB/s]

First-time prompt: Colab will ask "Allow this notebook to access your
Google secrets?" — click Grant access.

If it fails: Double-check that both secrets exist with the exact names
and that Notebook access is ON (Step 3).

------------------------------------------------------------------------
Cell 2 — Import Modules & Environment Probe
------------------------------------------------------------------------
Imports all libraries (torch, torchvision, PIL, mmap, multiprocessing,
etc.) with per-import timing, then probes the runtime environment.

Expected output (GPU):
  Importing modules...
    [0.10s] stdlib + PIL + multiprocessing primitives
    [3.50s] torch (core)
    [1.20s] torchvision (transforms + models)
    [0.01s] warnings + cudnn.benchmark flag
  CPU cores available: 2
  CUDA available: True
  GPU: Tesla T4
    [5.00s] CUDA probe

  Imports complete.

------------------------------------------------------------------------
Cell 3 — Dataset Path Helpers
------------------------------------------------------------------------
Defines helper functions that locate Train / Validation / Test folders
under /content/ regardless of how the zip was unpacked. Handles
uppercase, lowercase, and alternate name variants like training, valid,
dev, etc.

No output expected — these are function definitions only.

------------------------------------------------------------------------
Cell 4 — OSOptimizedImageDataset Class
------------------------------------------------------------------------
Defines the custom PyTorch Dataset with OS-level I/O optimizations:

  - os.scandir          faster directory traversal than os.listdir
  - _read_with_mmap     memory-mapped file reading via mmap.mmap,
                        os.open, os.fstat
  - _read_buffered      buffered I/O fallback with configurable
                        buffer size
  - Balanced subsampling to MAX_SAMPLES

No output expected.

------------------------------------------------------------------------
Cell 5 — MultiprocessDataCache Class
------------------------------------------------------------------------
Defines a thread-safe cache demonstrating OS synchronization primitives:

  - multiprocessing.Manager().dict()   shared memory dictionary visible
                                       across forked workers
  - multiprocessing.Lock()             mutex preventing race conditions
                                       on cache reads/writes
  - multiprocessing.Value('i', 0)      shared integer counters for
                                       cache hit/miss tracking

No output expected.

------------------------------------------------------------------------
Cell 6 — DeepfakeDetector Model
------------------------------------------------------------------------
Defines the full model: frozen ResNet18 backbone + trainable classifier
head.

  ResNet18 backbone  (frozen — ~11M params, no gradients computed)
      Classifier head  (trainable — ~132K params)
          Dropout(0.5)
          Linear(512 -> 256)
          ReLU
          Dropout(0.3)
          Linear(256 -> 2)   <-  output: Fake / Real

No output expected.

------------------------------------------------------------------------
Cell 7 — Training Function Definition
------------------------------------------------------------------------
Defines train_model_with_os_optimization(). This is the largest cell —
it only defines the function, it does not execute yet.

No output expected.

------------------------------------------------------------------------
Cell 8 — Run Training  <<< MAIN EXECUTION CELL >>>
------------------------------------------------------------------------
  model = train_model_with_os_optimization()

This triggers the full training pipeline in sequence:

  1. Prints the performance configuration (batch sizes, workers, AMP, device)
  2. Auto-detects dataset paths under /content/
  3. Loads Train and Validation datasets (up to 15,000 images each)
  4. Downloads pretrained ResNet18 weights (~45 MB, cached after first run)
  5. Extracts 512-dim feature vectors once over all images — the expensive step
  6. Frees the feature extractor from memory and clears the GPU cache
  7. Trains only the classifier head for 20 epochs on the cached features
  8. Saves the best checkpoint to best_deepfake_detector.pth

Expected output (GPU, ~15 seconds total):

  === Performance Configuration ===
  Max samples per split: 15000
  Extraction batch size: 512
  Head-training batch size: 128
  Epochs: 20 (only training the classifier head, so it's cheap)
  DataLoader workers: 2
  Pin memory: True
  Mixed precision (AMP): True

  Train dir: /content/Train
  Validation dir: /content/Validation

  === Loading Training Data ===
  Found 15000 images
  Fake: 7500, Real: 7500

  === Loading Validation Data ===
  Found 15000 images
  Fake: 7500, Real: 7500

  Loading pretrained ResNet18 (first time downloads ~45 MB, cached after)...

  === Extracting features (one-time cost) ===
      [train] 1500/15000 (10%)  2300 img/s  elapsed 0.7s
      [train] 3000/15000 (20%)  2280 img/s  elapsed 1.3s
      ...
    train: 15000 samples -> features (15000, 512) in 6.50s
    val:   15000 samples -> features (15000, 512) in 6.20s

  Classifier head trainable params: 132,610

  Epoch  1/20  train_loss=0.6821 train_acc=56.23%  val_loss=0.6540 val_acc=61.40%  (12.3 ms)
  Epoch  2/20  train_loss=0.6102 train_acc=67.11%  val_loss=0.5980 val_acc=68.73%  (11.8 ms)
  ...
  Epoch 20/20  train_loss=0.4231 train_acc=84.50%  val_loss=0.4580 val_acc=82.10%  (11.5 ms)

  ============================================================
  Training loop completed in 13.84s (process mem: 1240.3 MB)
  Best validation accuracy: 83.20%
  ============================================================

  Saved best model to best_deepfake_detector.pth

On CPU, feature extraction takes 2-3 minutes. The 20-epoch head
training is still under 1 second either way.

------------------------------------------------------------------------
Cell 9 — Evaluate Model
------------------------------------------------------------------------
  evaluate_model()

Loads best_deepfake_detector.pth, runs inference on the Test split,
and prints a full classification report.

Expected output:
  Test dir: /content/Test
  Found 15000 images
  Fake: 7500, Real: 7500

  Test Accuracy: 82.45%

  Confusion Matrix:
  [[6102 1398]
   [1224 6276]]

  Classification Report:
                precision    recall  f1-score   support
          Fake       0.83      0.81      0.82      7500
          Real       0.82      0.84      0.83      7500
      accuracy                           0.82     15000

================================================================================
7. EXPECTED OUTPUT & RESULTS
================================================================================

  Metric                       GPU (T4)        CPU only
  ---------------------------  --------------  --------------
  Feature extraction           10-15 seconds   2-3 minutes
  20-epoch head training       < 1 second      < 1 second
  Total wall time              ~15 seconds     ~2-3 minutes
  Validation accuracy          80-85%          80-85%
  Test accuracy                80-85%          80-85%
  Saved model file             best_deepfake_detector.pth

================================================================================
8. TROUBLESHOOTING
================================================================================

SecretNotFoundError: KAGGLE_USERNAME
  -> Open the Secrets panel in the left sidebar. Confirm both secrets exist,
     names are typed exactly as KAGGLE_USERNAME and KAGGLE_KEY, and Notebook
     access is ON for each one.

Colab asked "Grant access to secrets?" and I clicked Deny
  -> Go to Runtime -> Restart session, then re-run Cell 1. The permission
     prompt will appear again.

CUDA available: False after enabling T4 GPU
  -> Go to Runtime -> Restart session and re-run from Cell 1. The GPU is only
     recognized after a fresh session connects to the accelerated runtime.

"Could not locate Train and Validation folders under /content"
  -> The dataset unzipped into an unexpected subdirectory. Run this in a new
     cell to find where the images landed:

       import os
       for root, dirs, files in os.walk('/content'):
           if any(d.lower() in ['fake', 'real'] for d in dirs):
               print(root)

     If nothing prints, re-run Cell 1 to re-download the dataset.

torch.compile error
  -> Remove the line "feature_extractor = torch.compile(feature_extractor)"
     in Cell 7. This requires PyTorch 2.0+. Everything works correctly
     without it.

Out of GPU memory (CUDA out of memory)
  -> Reduce BATCH_SIZE from 512 to 256 or 128 in Cell 7. This only slows
     feature extraction slightly — it has no effect on accuracy.

Dataset download times out or stalls
  -> Re-run Cell 1. The Kaggle CLI picks up where it left off on a partial
     download.

================================================================================
9. PROJECT ARCHITECTURE REFERENCE
================================================================================

Notebook Cell Order

  Cell 1  -- Colab Secrets -> env vars -> pip install kaggle -> dataset download
  Cell 2  -- Imports + per-step timing + GPU/CPU environment probe
  Cell 3  -- Dataset path helpers (_find_dataset_splits, _has_fake_and_real, etc.)
  Cell 4  -- OSOptimizedImageDataset
               os.scandir        fast directory traversal (vs listdir)
               _read_with_mmap   mmap.mmap + os.open/fstat/close
               _read_buffered    open(..., buffering=N) fallback
  Cell 5  -- MultiprocessDataCache
               Manager().dict()  shared memory dict across forked workers
               Lock()            mutex for thread-safe cache access
               Value('i', 0)     shared hit/miss counters
  Cell 6  -- DeepfakeDetector (frozen ResNet18 backbone + classifier head)
  Cell 7  -- train_model_with_os_optimization()  [DEFINITION ONLY]
               Phase 1: Load datasets with OSOptimizedImageDataset
               Phase 2: Extract 512-dim features ONCE with frozen ResNet18
               Phase 3: Free extractor + torch.cuda.empty_cache()
               Phase 4: Train classifier head 20 epochs (ms/epoch)
               Phase 5: Save best model -> best_deepfake_detector.pth
  Cell 8  -- model = train_model_with_os_optimization()  [EXECUTES IT]
  Cell 9  -- evaluate_model()
               Load best_deepfake_detector.pth
               Run inference on Test split
               Accuracy + confusion matrix + classification report
