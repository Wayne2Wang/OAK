# Open Ad-hoc Categorization with Contextualized Feature Learning

**University of Michigan, UC Berkeley, Bosch Center for AI**

The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025

[Zilin Wang](https://wayne2wang.github.io/)\*, [Sangwoo Mo](https://sites.google.com/view/sangwoomo)\*, [Stella X. Yu](https://web.eecs.umich.edu/~stellayu/), [Sima Behpour](https://www.linkedin.com/in/sima-behpour-95037713b/), [Liu Ren](https://www.liu-ren.com/)

[[Paper](https://web.eecs.umich.edu/~stellayu/publication/doc/2025oakCVPR.pdf)] | [[Project Page](https://oak-cvpr2025.github.io/)] [[Poster](https://cvpr.thecvf.com/media/PosterPDFs/CVPR%202025/34699.png?t=1748965972.857557)] | [[Citation](#citation)]

![Main Image](assets/teaser.gif)

**TL;DR**: Ad-hoc categories are created dynamically to achieve specific tasks based on context at hand, such as things to sell at a garage sale. We introduce open ad-hoc categorization (OAK), a novel task requiring discovery of novel classes across diverse contexts, and tackle it by learning contextualized visual features with text guidance based on CLIP.


## Citation

If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.

```
@InProceedings{wang2025oakCVPR,
    author    = {Wang, Zilin and Mo, Sangwoo and Yu, Stella X. and Behpour, Sima and Ren, Liu},
    title     = {Open Ad-hoc Categorization with Contextualized Feature Learning},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {15108-15117}
}
```

## Installation

We develop this codebase on Python 3.12 and PyTorch 2.3.1 with CUDA 12.1

```
conda create -n oak python=3.12
conda activate oak
pip install -r requirements.txt
```

## Data Preparation

### Stanford *Action*, *Location*, *Mood*

We download the JPEG images from the official website of Stanford-40-Action [here](http://vision.stanford.edu/Datasets/40actions.html). 

The annotations for the *Action* context are parsed directly from the dataset. For the *Location* and *Mood* contexts, we parse from the annotations provided by [IC|TC](https://github.com/sehyunkwon/ICTC?tab=readme-ov-file#dataset-prep). You may directly download our parsed annotations from [here](https://drive.google.com/file/d/11x9eCmQXlYy059EIQLha-zCgbmNjaPLq/view?usp=sharing).

<pre>
Stanford40/
├── JPEGImages/
├── action.txt
├── location.txt
├── mood.txt
</pre>


### Clevr-4 *Texture*, *Color*, *Shape*, *Count*

The Clevr-4 datasets can be downloaded from the official website [here](https://www.robots.ox.ac.uk/~vgg/data/clevr4/). We use the 10k split.

<pre>
clevr_4_10k_v1/
├── images/
├── clevr_4_annots.json
</pre>


## Evaluation

Please download our provided models weights from [here](https://drive.google.com/file/d/1v3gi2_VIyszn5BO96dXcNioMm-6vCsr5/view?usp=sharing).

<pre>
weights/
├── oak_stanford_action.pt
├── oak_stanford_location.pt
├── oak_stanford_mood.pt
├── oak_clevr4_texture.pt
├── oak_clevr4_color.pt
├── oak_clevr4_shape.pt
├── oak_clevr4_count.pt
</pre>

To run evaluation, use the following commands:

```
python main.py [CONFIG_FILE] --eval_path [PATH_TO_WEIGHTS] --opts DATA.ROOT [PATH_TO_DATA_ROOT]
```


## Training

To run training, use the following commands. The training log and model weights will be automatically saved to the SAVE_DIR specified in the config file (default: ```saved```).

```
python main.py [CONFIG_FILE] --opts DATA.ROOT [PATH_TO_DATA_ROOT]
```

## License

OAK is released under the MIT License (refer to the LICENSE file for details).


## Acknowledgements

We would like to thank the following projects for their contributions to this work:

- [GCD](https://github.com/sgvaze/generalized-category-discovery)
- [CLIP](https://github.com/openai/CLIP)
- [VPT](https://github.com/KMnP/vpt)
- [DINO](https://github.com/facebookresearch/dino)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [TIMM](https://github.com/huggingface/pytorch-image-models)