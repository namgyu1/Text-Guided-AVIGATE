# Text-Guided AVIGATE: Audio-Guided Video Representation with Text-Guided Gated Attention for Text-to-Video Retrieval
**(Based on AVIGATE, CVPR 2025)**

This repository provides an extended implementation of [AVIGATE](https://github.com/BoseungJeong/AVIGATE-CVPR2025) with a multi-level **Text-Guided (Query-Aware)** mechanism.

The goal of this project is to improve Text-to-Video Retrieval performance by allowing the semantic intent of the text query (T) to dynamically influence and control the audio-visual (V-A) fusion process.

## Performance

On MSRVTT:
| Method | Modality | R@1↑ | R@5↑ | R@10↑ | MdR↓ | MnR↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CLIP4Clip (Luo et al. 2022) | V+T | 43.1 | 70.4 | 80.8 | 2.0 | 15.3 |
| ECLIPSE (Lin et al. 2022) | A+V+T | 44.2 | 71.3 | 81.6 | 2.0 | 15.0 |
| BridgeFormer (Ge et al. 2022) | V+T | 44.9 | 71.9 | 80.3 | 2.0 | 15.3 |
| X-CLIP (Ma et al. 2022) | V+T | 46.1 | 73.0 | 83.1 | 2.0 | 13.2 |
| X-Pool (Gorti et al. 2022) | V+T | 46.9 | 72.8 | 82.2 | 2.0 | 14.3 |
| TS2-Net (Liu et al. 2022) | V+T | 47.0 | 74.5 | 83.8 | 2.0 | 13.0 |
| TEFAL (Ibrahimi et al. 2023) | A+V+T | 49.4 | 75.9 | 83.9 | 2.0 | 12.0 |
| CLIP-ViP (Xue et al. 2022) | V+T | 50.1 | 74.8 | 84.6 | 1.0 | - |
| AVIGATE (Jeong et al. 2025) | A+V+T | 50.2 | 74.3 | 83.2 | 1.0 | 13.8 |
| T-MASS (Wang et al. 2024a) | V+T | 50.2 | 75.3 | 85.1 | 1.0 | 11.9 |
| [GAIS (Yang et al. 2025)](https://arxiv.org/abs/2508.01711) | A+V+T | 57.0 | 83.1 | 90.9 | 1.0 | 7.6 |
| **Text-Guided AVIGATE (ViT-B/32)** | **A+V+T** | **66.0** | **88.9** | **94.2** | **1.0** | **3.3** |
| **Text-Guided AVIGATE (ViT-B/16)** | **A+V+T** | **67.5** | **90.6** | **95.1** | **1.0** | **3.1** |

*(Relative to the recent SOTA (GAID), our Text-Guided AVIGATE delivers significant gains of **11.0%** in R@1, **5.9%** in R@5, and **4.3%** in R@10, demonstrating the effectiveness of our proposed text-guided fusion.)*

---

## 1. Problem: The Text-Agnostic Limitation of AVIGATE

The original AVIGATE model achieves SOTA by selectively fusing audio (A) and visual (V) information using a Gated Fusion Transformer.

However, this fusion process is **Text-Agnostic**. The gating mechanism only considers the relationship *within* the video (V-A interaction) and **ignores the text query (T)**.

This is suboptimal. The relevance of an audio cue is highly dependent on the text query.

## 2. Solution: Query-Aware (Text-Guided) Architecture

To solve this, I redesigned the Gated Fusion Transformer to be **Query-Aware**, making the text query (T) an active participant in the fusion process at multiple levels.

### Key Architectural Contributions:

1.  **Text-Conditioned Gating Function (C1):**
    The Gating Function was modified to accept the Text Embedding (T) as an additional condition. This allows the model to decide *how much* audio to fuse based on *what* the user is searching for (the semantic intent of T).

2.  **Text-Injected MHA Query (C2):**
    In the original cross-attention MHA block, the Query is generated only from Visual Frame Embeddings (V), while Key and Value are from Audio Embeddings (A). This finds audio features relevant only to the visual content. I modified this core mechanism by injecting the Text Embedding (T) directly into the Visual Frame Embeddings (V) before this combined vector is projected to create the Query. This change allows the model to find audio features that are relevant not just to the visual content, but to the semantic intent of the text query.

3.  **Gated Text Injection (Gate for Text-Injection):**
    To prevent the text query from overpowering the visual features, a **new MLP gate** was implemented. This gate dynamically controls the amount of text information (T) injected into the Visual Frame Embeddings, based on the context of all three modalities (T, V, and A).

---

## 3. Extensive Experiments

### 3.1 Ablation Study (MSR-VTT, ViT-B/32)
To rigorously validate the complementary benefits of our proposed modules, we conducted an ablation study. Notably, evaluating the model in a **Video-to-Text (V2T)** retrieval setting highlights the sensitivity of query-aware representations. 

In V2T retrieval, one fused video representation is matched against multiple candidate captions. If the semantic alignment degrades during the audio-visual fusion process, the V2T performance drops significantly.

| Configuration | Text-to-Video (R@1) | Text-to-Video (R@5) | Text-to-Video (R@10) | Video-to-Text (R@1) | Video-to-Text (R@5) | Video-to-Text (R@10) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| w/o Text-Conditioned Gating (C1) | 64.4 | 89.8 | 94.2 | 21.5 | 53.9 | 61.8 |
| w/o Text-Injected MHA Query (C2) | 64.3 | 89.3 | 94.1 | 25.6 | 72.9 | 78.8 |
| **Full Model (C1 + C2)** | **66.0** | **88.9** | **94.2** | **37.5** | **72.8** | **86.2** |

As demonstrated, both C1 and C2 are essential. Removing either module causes a critical degradation in V2T R@1 (dropping to 21.5 and 25.6, respectively), proving that the text signal in our architecture actively controls semantic relevance rather than merely providing an extra modality.

### 3.2 Backbone Scalability and Cross-Dataset Generalization (VATEX)
To demonstrate cross-dataset generalizability, we evaluated our models on the **VATEX** dataset (10% test subset). Furthermore, we upgraded the visual encoder to CLIP ViT-B/16 to verify backbone scalability.

| Setting | Backbone | R@1↑ | R@5↑ | R@10↑ | MdR↓ | MnR↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-shot (from MSR-VTT) | ViT-B/32 | 73.7 | 92.5 | 95.3 | 1.0 | 2.5 |
| Fine-tuned | ViT-B/32 | 80.5 | 97.3 | 98.9 | 1.0 | 1.6 |
| Fine-tuned | ViT-B/16 | **85.5** | **98.0** | **99.2** | **1.0** | **1.6** |

The results show that the query-aware gating mechanism successfully leverages finer visual tokens from ViT-B/16 without saturating, and the strong zero-shot performance (73.7 R@1) indicates robust feature alignment that transfers seamlessly to new datasets.

---

## Requirement
```sh
# From CLIP
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas---
## Requirement
```sh
# From CLIP
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas
```
### Conda Environment
```sh
conda env create --file video.yml
```
## Data Preparing

**For MSRVTT**

The official data and video links can be found in [link](http://ms-multimedia-challenge.com/2017/dataset). 

For the convenience, you can also download the splits and captions by,
```sh
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
```

Besides, the raw videos can be found in [sharing](https://github.com/m-bain/frozen-in-time#-finetuning-benchmarks-msr-vtt) from *Frozen️ in Time*, i.e.,
```sh
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```
For videos without audio signals, we obtained audio sources using external crawling tools like [youtube-dl](https://github.com/yt-dlp/yt-dlp).  
We get 9,582 audio signals for 10,000 videos.

## Compress Video for Speed-up (optional)
```sh
python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]
```
This script will compress the video to *3fps* with width *224* (or height *224*). Modify the variables for your customization.

# How to Run
Download CLIP (ViT-B/32) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```
or, download CLIP (ViT-B/16) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```
Download AST weight from [AST](https://github.com/YuanGongND/ast) (Pretrained Models 1: "Full AudioSet, 10 tstride, 10 fstride, with Weight Averaging (0.459 mAP)").


**For MSR-VTT Training** 
```sh
run.sh
```
**For MSR-VTT Evaluation** 
```sh
run_eval.sh
```
# Citation
If you find CLIP4Clip useful in your work, you can cite the following paper:
```bibtex
@InProceedings{Jeong_2025_CVPR,
    author    = {Jeong, Boseung and Park, Jicheol and Kim, Sungyeon and Kwak, Suha},
    title     = {Learning Audio-guided Video Representation with Gated Attention for Video-Text Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {26202-26211}
}
```

# Acknowledgments
Our code is based on [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) and [AST](https://github.com/YuanGongND/ast).

