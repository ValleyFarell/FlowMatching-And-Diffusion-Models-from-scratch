# Flow Matching and Diffusion Models

Educational PyTorch implementations of Flow Matching, score matching, and continuous-time diffusion, with training curves, generated samples, and particle trajectories.

- **Two Moons:** independently trained velocity, score, and noise-prediction networks.
- **MNIST:** one shared denoiser converted into velocity, score, and noise predictions, with ODE and SDE sampling. This uses a reweighted denoising objective rather than the direct objectives in Algorithm 4.

## Usage

Open `FlowMatchingAndDiffusionModels_MNIST.ipynb` in Google Colab or Jupyter and run the cells in order. A GPU is recommended.

```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn tqdm
```

Optionally place `MNISTDenoiser.pt` in the working directory to load pretrained MNIST weights; otherwise, the notebook trains the denoiser.

## Lecture Source

Based on [An Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/), MIT. See the lecture notes, particularly Sections 3–4, for the underlying theory and algorithms.
