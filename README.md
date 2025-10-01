

# LoRA-Driven Anime Style Generation

This repository provides a comprehensive Jupyter Notebook for fine-tuning **Stable Diffusion v1.5** using **Low-Rank Adaptation (LoRA)** to generate images in various anime art styles. The notebook includes a full implementation of training routines, hyperparameter exploration, and a comparative analysis against **Textual Inversion** and **DreamBooth**. It also explores style blending through LoRA weight interpolation.

All experiments, code, evaluation metrics, and generated figures are self-contained within the notebook. When executed, it automatically generates all necessary outputs, including model checkpoints, sample images, and performance logs.

-----

##  Key Features

  * **Multi-Style Adaptation**: Fine-tune Stable Diffusion on distinct artistic styles, including **Ghibli**, **Makoto Shinkai**, and **American Comic**, to test the generalization capabilities of LoRA.
  * **Hyperparameter Analysis**: Systematically investigate the impact of key hyperparameters such as LoRA rank, training steps, and learning rate to identify optimal configurations.
  * **Comparative Benchmarking**: Evaluate LoRA's performance against Textual Inversion and DreamBooth, focusing on image quality, resource consumption, and model footprint.
  * **Creative Style Interpolation**: Explore the potential of linearly interpolating LoRA weights from different styles to create novel and coherent hybrid aesthetics.

-----

##  Repository and Data Structure

The project is organized into a primary notebook and a clear data directory structure.

### File Layout

```
.
├── LoRA_AnimeStyle_Final.ipynb  # The main Jupyter Notebook for all experiments.
└── README.md
```

### Expected Data Layout

The notebook expects training data and will generate all outputs in the following structure. Paths are configurable.

```
SD/
├── Datasets/
│   ├── ghibli/                  # Training images (.png, .jpg) for Style 1
│   │   └── (optional captions).txt
│   ├── shinkai/                 # Training images for Style 2
│   └── comic/                   # Training images for Style 3
│   └── prompts/
│       └── validation_prompt.txt # Custom validation prompts
│
└── runs/                        # All generated outputs are saved here
    └── <timestamped_experiment_folder>/
        ├── checkpoints/         # Saved LoRA adapter weights
        ├── inference/           # Generated images
        ├── thumbnails/          # Image strips for quick comparison
        ├── config.json          # Experiment configuration
        ├── metrics.json         # Evaluation results
        └── logs.txt             # Training logs
```

-----

##  Getting Started

### Requirements

  * **Python 3.10+**
  * **NVIDIA GPU** with CUDA support (A100 or equivalent recommended).
  * Key libraries: `torch`, `diffusers`, `transformers`, `peft`, `accelerate`.

You can install the required packages using pip:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core diffusion and LoRA libraries
pip install diffusers[torch] transformers peft accelerate pillow tqdm

# Install optional packages for advanced evaluation metrics
pip install clean-fid torch-fidelity lpips
```

### Data Preparation

1.  Create a directory for your datasets (e.g., `SD/Datasets/`).
2.  Inside, create a sub-folder for each art style (e.g., `ghibli/`, `shinkai/`).
3.  Place your training images (`.png`, `.jpg`, etc.) inside the respective style folders.
4.  **(Optional)** For captioned training, add `.txt` files with the same name as each image. If no caption is provided, a default one will be used.
5.  **(Optional)** Create a `prompts/validation_prompt.txt` file to specify custom prompts for inference. If this file is missing, the notebook will generate a default set.

### Running the Notebook

#### Option A: Kaggle

The notebook is ready to run on Kaggle. Simply open the link and execute the cells in order.

  * **Kaggle Notebook Link:** [CVFinal-Group10](https://www.kaggle.com/code/popcatty/cvfinal-group10)

#### Option B: Local Environment

1.  Clone this repository.
2.  Install the dependencies as described above.
3.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook LoRA_AnimeStyle_Final.ipynb
    ```
4.  Run the cells sequentially. The notebook will handle all training, inference, and evaluation, saving the results in a timestamped folder under `runs/`.

-----

##  Evaluation and Metrics

The notebook automatically evaluates model performance using a suite of standard metrics.

  * **CLIP Score**: Measures the semantic similarity between the generated image and the text prompt (enabled by default).
  * **FID, IS, KID**: Optional metrics for assessing image quality and distribution similarity. Enable them by installing `clean-fid` and `torch-fidelity`.
  * **LPIPS**: Optional metric for measuring the diversity of generated images. Enable it by installing the `lpips` package.
  * **Baseline Comparison**: All metrics are automatically compared against the base Stable Diffusion v1.5 model (without any fine-tuning) to quantify the impact of LoRA.

-----

##  Reproducing the Study

The notebook is structured into four distinct experiments. You can run them in order to reproduce the original study:

1.  **Experiment I**: Train individual LoRA adapters for the Ghibli, Shinkai, and American Comic styles.
2.  **Experiment II**: Run an ablation study to analyze the effects of different hyperparameters (rank, steps, learning rate).
3.  **Experiment III**: Conduct a comparative analysis of LoRA, Textual Inversion, and DreamBooth under identical conditions.
4.  **Experiment IV**: Interpolate the weights of two trained LoRA adapters to generate and analyze hybrid-style images.

-----
##  Experimental Results

A compressed archive containing all experimental results, including generated images, logs, and metrics, can be downloaded from the following link:

[**Download Experimental Results from Google Drive**](https://drive.google.com/drive/folders/1W-anKsWulkiMmBSZ1pfbLtyAe6p3Oz88?usp=sharing)
-----
##  Additional Notes

  * **Mixed Precision**: The notebook defaults to `bf16` precision on compatible hardware (e.g., Ampere GPUs) and falls back to `fp16` otherwise.
  * **Memory Optimization**: To ensure stability on systems with limited VRAM, several optimizations are enabled by default, including gradient checkpointing, SNR-weighted loss calculation, attention slicing, and VAE tiling.
  * **Reproducibility**: All configurations and metric results are saved to JSON files, ensuring that experiments can be easily tracked and reproduced.

  ## References

### Code Repositories
- Hoper-J / [AI-Guide-and-Demos-zh_CN](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN)  
  *Reference for LoRA fine-tuning, esp. “14b” simplified notebook. Used as structural inspiration (~10–15% overlap); majority of this repo’s code has been newly extended and reorganized.*

- Hugging Face / [diffusers](https://github.com/huggingface/diffusers)  
  *Core library for Stable Diffusion, pipelines, UNet, VAE, and schedulers.*

- Hugging Face / [transformers](https://github.com/huggingface/transformers)  
  *Provides CLIP text encoder and tokenizer, plus general NLP utilities.*

- Hugging Face / [peft](https://github.com/huggingface/peft)  
  *LoRA integration and parameter-efficient fine-tuning toolkit.*

- Hugging Face / [accelerate](https://github.com/huggingface/accelerate)  
  *Optional library for efficient multi-GPU/mixed precision training.*

---

### Articles & Guides
- *Using LoRA to Fine-tune Stable Diffusion: Opening the Furnace and Implementing Your First AI Drawing.*  
  *Conceptual reference for methodology and experimental design.*

- Hugging Face blog: *[Parameter-Efficient Fine-Tuning (PEFT): LoRA and Beyond](https://huggingface.co/blog/peft)*  
  *Provided key insights on efficient adaptation methods for large models.*

- Papers/Community Notes on Stable Diffusion & LoRA (various).  
  *General inspiration for training design, evaluation metrics, and best practices.*
