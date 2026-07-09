# MLP Lunar Lander — Neural Network Control from Scratch

A multi-layer perceptron built entirely from first principles — no NumPy, scikit-learn, or TensorFlow — trained via manual backpropagation to autonomously land a simulated rocket. Built for the CE889 Neurocomputing module (University of Essex).

## Context

The game/simulation itself — `Main.py`, `GameLoop.py`, `Lander.py`, `Surface.py`, `Vector.py`, `CollisionUtility.py`, `Controller.py`, `EventHandler.py`, `MainMenu.py`, `ResultMenu.py`, `DataCollection.py` — was provided as course scaffolding (credited in `Main.py` to Lewis Veryard and Hugo Leon-Garza). My work is the neural network layer built on top of it: the from-scratch MLP and backpropagation implementation, the training pipeline, the mathematical controller used to generate clean training labels, and the hyperparameter search.

## What I built

- **A from-scratch MLP** (`lander_mlp_neuron_based.py`) — a `Neuron` class handles the forward pass and activation derivative; an `MLP` class wires neurons into layers and implements the full backpropagation update rule (output-layer delta → hidden-layer delta → gradient descent weight update) directly from the underlying calculus, with no ML library doing the differentiation.
- **Architecture: 2 → 6 → 2** — 2 inputs (normalized x/y distance to the landing pad), 6 hidden neurons with sigmoid activation, 2 outputs (normalized horizontal/vertical velocity) with linear activation.
- **A mathematical controller for training labels** — the raw human-gameplay velocities were too noisy to learn from well, so they were replaced with velocities computed by a proportional controller (horizontal speed scales with distance from the pad; target altitude scales with horizontal distance, to clear obstacles) and used as the regression targets instead.
- **Hyperparameter selection** — backed by literature (Heaton 2017, Gage 2011, Goodfellow/Bengio/Courville's *Deep Learning*, LeCun's *Efficient Backprop*) and empirical testing in MATLAB (`hyperparams.m`), comparing hidden-layer sizes (4/6/8/10) and learning rates (0.001–0.12). Settled on **6 hidden neurons, learning rate 0.03**.
- **`NeuralNetHolder.py`** — the bridge that loads the trained weights back into the live game and converts network outputs into real velocities each frame, using the same normalization scheme as training.

## Results

| Split | Samples | MSE |
|---|---|---|
| Training | 60,337 | — |
| Validation | 12,929 | 0.001498 |
| Test | 12,930 | 0.001493 |

Trained for 100 epochs using online (per-sample) gradient descent, no momentum, weights initialized via `random.uniform(-1, 1)` to break symmetry, and a fixed seed for reproducibility.

## Repository structure

```
mlp-lunar-lander/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── MLP_Backpropagation.pdf   # full write-up: architecture rationale, hyperparameter comparisons, sources
├── Files/
│   └── Config.con                # screen size, asset paths, data-collection flag
├── Sprites/
│   ├── rocket_lander.png         # lander sprite + window icon
│   └── BackGround.jpg            # background (see note below on format)
├── Main.py
├── GameLoop.py
├── GameLogic.py
├── Lander.py
├── Surface.py
├── Vector.py
├── CollisionUtility.py
├── Controller.py
├── EventHandler.py
├── MainMenu.py
├── ResultMenu.py
├── DataCollection.py
├── lander_mlp_neuron_based.py    # from-scratch MLP + training
├── NeuralNetHolder.py            # loads trained weights into the game
├── lander_normalization_code.py  # min-max normalization utility
├── lander_neuron_weights.txt     # trained weights
└── hyperparams.m                 # MATLAB hyperparameter comparison
```

Not committed: `ce889_dataCollection.csv` and its normalized counterpart — large, and easy to regenerate by playing the game in Data Collection mode.

## Setup & running

Needs `Files/Config.con` plus the assets it points to in `Sprites/`. One change from the original: `BACKGROUND_IMG_PATH` now points at `BackGround.jpg` instead of `BackGround.bmp` — same 1920×1080 image, but the `.bmp` was 6.2MB uncompressed versus 57KB as a `.jpg`, which isn't worth carrying in a git repo. Pygame loads both identically.

```bash
pip install -r requirements.txt
python Main.py
```

From the main menu:
- **Play Game** — manual control
- **Data Collection** — records gameplay to `ce889_dataCollection.csv`
- **Neural Network** — lets the trained MLP fly the lander autonomously

To retrain from scratch:

```bash
python lander_mlp_neuron_based.py
```

Reads `ce889_dataCollection.csv`, trains the MLP, and writes updated weights to `lander_neuron_weights.txt`.

## Tech stack

Python (pure, for the network itself) · Pygame · MATLAB (hyperparameter comparison)
