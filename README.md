# Healthcare Image–Text Multimodal (Script)

A concise Python script for a healthcare **image–text multimodal** pipeline (e.g., chest X-ray + report text). 

## Quickstart

```bash
# (Optional) create env
conda create -n med-mm python=3.10 -y
conda activate med-mm

# install deps (adjust torch to your CUDA/OS)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers scikit-learn pillow pandas numpy tqdm matplotlib
# optional extras
pip install albumentations opencv-python sentencepiece
