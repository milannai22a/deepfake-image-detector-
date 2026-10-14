# Deepfake Detector

A computer vision project that decides whether a face photo is **real or AI-generated**, and shows **where** on the face the model found suspicious signs.

Everything lives in one Google Colab notebook (`deepfake_detector_colab.ipynb`). Running it writes the full project to disk, tests it, downloads a dataset, trains the model, shows results, and launches a web demo.

---

## What it does

- Takes a face photo (or video frames) and outputs a **fake probability** between 0 and 1.
- Draws a **Grad-CAM heatmap** showing which face region (eyes, mouth/jaw, forehead edge, nose/cheeks) influenced the decision.
- Gives a short plain-language note, e.g. "Likely manipulated - the strongest signal is around the mouth / jaw area."
- Offers a **Gradio web demo** with a public link, so anyone can upload a photo and test it.

## How it works

```
Image / video frame
        |
Face detection + crop (OpenCV Haar cascade, 224x224)
        |
   +----+--------------------------+
   |                               |
Spatial branch                Frequency branch
EfficientNet-B0               FFT log-magnitude spectrum
(texture / edge artifacts)    -> small CNN (generator fingerprints)
   |                               |
   +--------------+----------------+
                  |
           Fusion classifier
                  |
          P(fake)  +  Grad-CAM heatmap  +  explanation
```

Using two branches is deliberate. A spatial-only CNN can be fooled once compression smooths out pixel artifacts, while a frequency signature may still survive, and the reverse can also happen.

Training includes augmentation that imitates real-world "laundering" (JPEG compression, blur, down/up-scaling, flips), so the model sees imperfect images like those found on social media.

---

## Project structure (created by the notebook)

```
deepfake_detector/
├── dfd/
│   ├── __init__.py
│   ├── preprocessing.py   # face detection, cropping, normalisation, video frame extraction
│   ├── model.py            # dual-branch DeepfakeDetector + save/load checkpoint helpers
│   ├── dataset.py          # FaceCropDataset (real=0, fake=1) + augmentation
│   ├── prepare_data.py     # builds data/train and data/val from a downloaded dataset
│   ├── train.py            # training loop, metrics, early stopping, optional OOD test
│   ├── explain.py          # Grad-CAM + region interpretation + overlay
│   └── pipeline.py         # single entry point: analyze_image / analyze_video
├── app/
│   └── gradio_app.py       # web demo
├── tests/
│   └── smoke_test.py       # offline end-to-end check using synthetic data
├── models/                 # trained weights land here (detector_best.pt + .json)
└── data/                   # train/ and val/ folders created by prepare_data
```

---

## Quick start (Google Colab)

1. Open the notebook in [Google Colab](https://colab.research.google.com).
2. **Runtime -> Change runtime type -> T4 GPU**.
3. **Runtime -> Run all**.

The notebook runs these stages in order:

| Step | What happens |
|---|---|
| 0. Install | Installs `timm`, `grad-cam`, `kagglehub`, `gradio`, `scikit-learn`. Torch and OpenCV are already in Colab and are not touched. |
| 1. Write files | `%%writefile` cells create every file in the structure above. |
| 2. Smoke test | Runs `tests/smoke_test.py` (about 1 minute) to confirm every module works before the long run. |
| 3. Download data | Downloads **140k Real and Fake Faces** via `kagglehub`. |
| 4. Prepare data | Builds `data/train` and `data/val` using symlinks (no copying or re-encoding). |
| 5. Train | Trains for 10 epochs (roughly 10-20 minutes on a T4 for 6,000 training images). |
| 6. Inspect | Prints best metrics and shows Grad-CAM on real and fake validation images. |
| 7. Demo | Launches the Gradio app with a public `gradio.live` link. |
| 8. Save (optional) | Copies the trained model to Google Drive. |

### Save your model before the session ends

Colab wipes its storage when the session ends. If you do not save `models/detector_best.pt`, you will have to train again. Either download it from the Files panel, or run:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp models/detector_best.pt models/detector_best.json /content/drive/MyDrive/
```

---

## Dataset

**Default:** [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces). Real images are FFHQ photos; fake images are StyleGAN-generated.

If `kagglehub` asks for credentials: Kaggle -> Settings -> Create New Token, then in a notebook cell:

```python
import os
os.environ["KAGGLE_USERNAME"] = "your_username"
os.environ["KAGGLE_KEY"] = "your_key"
```

**Your own data:** skip the download cell and set `DATA_SRC` to a folder containing:

```
<DATA_SRC>/
├── train/  real/  fake/
└── valid/  real/  fake/
```

`prepare_data` searches the folder tree automatically, so the exact nesting does not matter. For raw photos or video frames (not already face crops), add `--detect` to find and crop faces first:

```bash
python -m dfd.prepare_data --src /path/to/data --out data --n_train 3000 --n_val 600 --detect
```

---

## Training options

```bash
python -m dfd.train \
    --train_dir data/train --val_dir data/val \
    --epochs 10 --batch_size 32 --out models/detector_best.pt
```

| Flag | Default | Meaning |
|---|---|---|
| `--backbone` | `efficientnet_b0` | Any `timm` model name |
| `--epochs` | 10 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 3e-4 | Learning rate (AdamW, cosine schedule) |
| `--patience` | 4 | Early stopping; `0` disables it |
| `--ood_dir` | none | Folder of images from a *different* generator, to measure the generalisation gap |
| `--no_pretrained` | off | Skip ImageNet weights (for offline tests) |
| `--seed` | 42 | Random seed |

Training uses mixed precision on GPU, class-imbalance weighting, and gradient clipping. It saves the best checkpoint by validation AUC, plus a `detector_best.json` file with per-epoch loss, accuracy, precision, recall, F1 and AUC.

---

## Using the trained model

```python
import cv2
from dfd.pipeline import DeepfakePipeline

pipe = DeepfakePipeline("models/detector_best.pt")

# Image
img = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
res = pipe.analyze_image(img)
if res["face_found"]:
    print(f"P(fake) = {res['probability_fake']:.2f}")
    print(res["top_region"], "-", res["note"])
    # res["face"], res["overlay"] are images you can display or save

# Video (samples every Nth frame, up to max_frames)
vid = pipe.analyze_video("clip.mp4", every_n=15, max_frames=32)
print(vid["mean_prob"], vid["max_prob"])
```

To run the web demo outside the notebook:

```bash
python app/gradio_app.py
```

Using the model later on exam or demo day does **not** require retraining. Upload the project files and `detector_best.pt`, then run only the install and load/inference cells.

---

## Running on your own machine

Colab already provides torch and OpenCV. On your own machine, install them as well:

```bash
pip install torch torchvision opencv-python-headless numpy pillow \
            timm grad-cam kagglehub gradio scikit-learn matplotlib
python tests/smoke_test.py
```

---

## Limitations

- **Trained on one generator family.** The default dataset uses StyleGAN fakes. Newer diffusion-based fakes leave different traces, so expect lower accuracy on them. Use `--ood_dir` to measure this honestly.
- **Compression and re-uploads.** Augmentation helps, but heavy platform compression can still hide artifacts.
- **Face detector.** Haar cascade is lightweight but weaker on angled, small, or low-light faces. It takes only the largest face per frame.
- **Grad-CAM is a hint, not proof.** The heatmap shows where the model looked, not a guaranteed manipulation location. A model that is untrained or weakly trained produces meaningless heatmaps.
- **No audio.** Voice cloning is not covered.
- **Not a verdict.** Treat the score as a signal for further checking, especially for high-stakes decisions.

## Troubleshooting

| Problem | Fix |
|---|---|
| `GPU: NONE` | Runtime -> Change runtime type -> T4 GPU, then rerun |
| Kaggle download asks for credentials | Set `KAGGLE_USERNAME` and `KAGGLE_KEY` (see Dataset) |
| `Could not find a train/{real,fake} folder` | Check `DATA_SRC` contains `train/real`, `train/fake`, and `valid` (or `val`/`test`) with `real`/`fake` inside |
| `No face detected` | Use a clearer, front-facing, well-lit photo |
| Demo says "untrained model" | `models/detector_best.pt` is missing; train first or upload the saved file |
| Weights lost after reconnecting | Colab storage is temporary; save to Drive or download the file (see above) |
| cv2 errors after installing packages | Do not uninstall or reinstall OpenCV in Colab; use the version already installed |

## Tech stack

PyTorch, timm (EfficientNet-B0), OpenCV, pytorch-grad-cam, scikit-learn, Gradio, kagglehub.
