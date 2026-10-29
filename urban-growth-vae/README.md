# AI-Powered Urban Expansion Prediction and City Growth Simulation using VAE

An MCA project for urban satellite imagery modeling, spatial layout learning, and generative growth simulation using Convolutional Variational Autoencoders (ConvVAE).

## Project Structure

```text
urban-growth-vae/
├── data/
│   ├── raw/             # Original Sentinel-2 satellite imagery (EuroSAT RGB)
│   └── processed/       # Resized (128x128), normalized, clean RGB images
├── models/
│   └── vae_best.pth     # Best trained ConvVAE checkpoint (PyTorch weights)
├── outputs/
│   ├── loss_curve.png   # Training vs validation loss curve
│   ├── reconstructions/ # Side-by-side original vs VAE reconstructed images
│   └── generated/       # Generated urban layout scenario images & comparison
├── scripts/
│   ├── download_data.py # Dataset downloader script
│   ├── preprocess.py    # Preprocessing pipeline
│   ├── model.py         # ConvVAE architecture & SatelliteDataset loader
│   ├── train.py         # Training pipeline (Adam, loss logging, checkpointing)
│   └── generate.py      # Scenario generation and side-by-side visualization
├── venv/                # Python virtual environment
├── requirements.txt     # Dependencies for the project
└── README.md            # Project documentation
```

## Setup Instructions (Windows PowerShell)

### 1. Navigate to the project directory
```powershell
cd urban-growth-vae
```

### 2. Activate the virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays script execution restriction, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

### 3. Install required libraries
```powershell
pip install -r requirements.txt
```

---

## Running the Project (Phases 1 to 6)

### Phase 2: Download Dataset
```powershell
python scripts/download_data.py
```

### Phase 3: Preprocess Images (128x128 RGB)
```powershell
python scripts/preprocess.py
```

### Phase 4: Test Model Architecture & Forward Pass
```powershell
python scripts/model.py
```

### Phase 5: Train the ConvVAE Model
```powershell
python scripts/train.py --epochs 25 --batch-size 8
```
This will:
- Split the dataset into 80% train and 20% validation.
- Train the model using Reconstruction Loss + KL Divergence.
- Save the best checkpoint to `models/vae_best.pth`.
- Save reconstruction sample images to `outputs/reconstructions/`.
- Save the loss plot to `outputs/loss_curve.png`.

### Phase 6: Generate Urban-Growth Scenarios
```powershell
python scripts/generate.py --num-scenarios 3
```
This will:
- Load the trained checkpoint from `models/vae_best.pth`.
- Encode an input satellite image into latent space.
- Generate alternative structural urban layout scenarios.
- Save individual images to `outputs/generated/`.
- Display a side-by-side comparison popup window.

---

## Scientific Assessment & Temporal Limitations

- **Current Capability:** The trained ConvVAE learns the continuous spatial probability distribution of urban satellite imagery. The generated scenarios represent **morphological generative variations and synthetic urban layouts** on this learned spatial manifold.
- **Temporal Constraint:** The current EuroSAT dataset consists of individual static image patches rather than registered multi-temporal time-series pairs (e.g., Year 2015 vs. Year 2025 of the exact same city coordinates). Therefore, outputs are **generative layout simulations**, not chronological ground-truth future predictions.
- **Roadmap for Genuine Temporal Prediction:**
  To simulate true historical-to-future urban expansion, the project requires paired temporal imagery $(T_0, T_1)$ and a **Conditional VAE (CVAE)** where $T_0$ is fed as a conditioning tensor to predict $T_1$.
