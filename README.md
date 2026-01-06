Clone the project
Clone these videos from here: https://drive.google.com/drive/folders/1Bi0oE8lZ8sXPlAACdOFnsC0fWA6KUZUr?usp=sharing

I use uv package manager btw, cuz its made in rust 

so do this

```
uv sync
```
and your done! Virtual environment is done!

now run 
```
uv run main.py
```
this runs main.py with uv's virtual environment python so its so streamlined and smooth

there might be some errors with timm module and make sure its `1.0.22` not `1.0.23`

I have not tried train.py yet because the dataaset using ground model isn't working as ai prompt engineering sucks for me as I cannot describe the game pieces to ai properly...

uv pip list (in a working machine)
```
Package                  Version
------------------------ -----------
addict                   2.4.0
asttokens                3.0.1
autodistill              0.1.29
autodistill-grounded-sam 0.1.2
certifi                  2026.1.4
charset-normalizer       3.4.4
click                    8.3.1
comm                     0.2.3
contourpy                1.3.3
cycler                   0.12.1
decorator                5.2.1
defusedxml               0.7.1
executing                2.2.1
filelock                 3.20.2
filetype                 1.2.0
fonttools                4.61.1
fsspec                   2025.12.0
hf-xet                   1.2.0
huggingface-hub          0.36.0
idna                     3.7
ipython                  9.8.0
ipython-pygments-lexers  1.1.1
ipywidgets               8.1.8
jedi                     0.19.2
jinja2                   3.1.6
joblib                   1.5.3
jupyterlab-widgets       3.0.16
kiwisolver               1.4.9
markupsafe               3.0.3
matplotlib               3.10.8
matplotlib-inline        0.2.1
mpmath                   1.3.0
networkx                 3.6.1
numpy                    1.26.4
nvidia-cublas-cu12       12.8.4.1
nvidia-cuda-cupti-cu12   12.8.90
nvidia-cuda-nvrtc-cu12   12.8.93
nvidia-cuda-runtime-cu12 12.8.90
nvidia-cudnn-cu12        9.10.2.21
nvidia-cufft-cu12        11.3.3.83
nvidia-cufile-cu12       1.13.1.3
nvidia-curand-cu12       10.3.9.90
nvidia-cusolver-cu12     11.7.3.90
nvidia-cusparse-cu12     12.5.8.93
nvidia-cusparselt-cu12   0.7.1
nvidia-nccl-cu12         2.27.5
nvidia-nvjitlink-cu12    12.8.93
nvidia-nvshmem-cu12      3.3.20
nvidia-nvtx-cu12         12.8.90
opencv-python            4.11.0.86
opencv-python-headless   4.10.0.84
packaging                25.0
pandas                   2.3.3
parso                    0.8.5
pexpect                  4.9.0
pi-heif                  1.1.1
pillow                   12.1.0
pillow-avif-plugin       1.5.2
platformdirs             4.5.1
prompt-toolkit           3.0.52
psutil                   7.2.1
ptyprocess               0.7.0
pure-eval                0.2.3
py-cpuinfo               9.0.0
pycocotools              2.0.11
pygments                 2.19.2
pyparsing                3.3.1
python-dateutil          2.9.0.post0
python-dotenv            1.2.1
pytz                     2025.2
pyyaml                   6.0.3
regex                    2025.11.3
requests                 2.32.5
requests-toolbelt        1.0.0
rf-groundingdino         0.3.0
rf-segment-anything      1.0
roboflow                 1.2.11
safetensors              0.7.0
scikit-learn             1.8.0
scipy                    1.16.3
seaborn                  0.13.2
setuptools               80.9.0
six                      1.17.0
stack-data               0.6.3
supervision              0.24.0
sympy                    1.14.0
threadpoolctl            3.6.0
timm                     1.0.22
tokenizers               0.22.1
torch                    2.9.1
torchvision              0.24.1
tqdm                     4.67.1
traitlets                5.14.3
transformers             4.57.3
triton                   3.5.1
typing-extensions        4.15.0
tzdata                   2025.3
ultralytics              8.3.3
ultralytics-thop         2.0.18
urllib3                  2.6.2
wcwidth                  0.2.14
widgetsnbextension       4.0.15
yapf                     0.43.0
```
