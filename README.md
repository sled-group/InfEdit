# Inversion-Free Image Editing with Natural Language


### [Project Page](https://sled-group.github.io/InfEdit/) | [Paper](https://arxiv.org/abs/2312.04965) | [🤗Demo🔥](https://huggingface.co/spaces/sled-umich/InfEdit) | [Handbook](https://github.com/sled-group/InfEdit/tree/website)

Sihan Xu*, Yidong Huang*, Jiayi Pan, Ziqiao Ma, Joyce Chai  
University of Michigan, Ann Arbor  
University of California, Berkeley

![icon](infedit_gif.gif)

## Setup
This code was tested with python 3.9, [Pytorch](https://pytorch.org/) 2.2.1 using pretrained models through [huggingface / diffusers](https://github.com/huggingface/diffusers#readme). Specifically, we implemented our method over [LCM](https://huggingface.co/docs/diffusers/api/pipelines/latent_consistency_models) and [prompt2prompt](https://github.com/google/prompt-to-prompt). 

## Quickstart

### Requirements
We implemented our method with [diffusers](https://github.com/huggingface/diffusers#readme) with a similar code structure to [prompt2prompt](https://github.com/google/prompt-to-prompt) but we modified the code to adapt to newest diffusors version. You can download the requirements using 
```base
pip install -r requirements.txt
```


### Demos




**Online demos**

We provide online [demo](https://huggingface.co/spaces/sled-umich/InfEdit) with Gradio app. You can play with the it or clone it to your own device!

**Local Gradio demo**

You can launch the provided Gradio demo locally with

```bash
python app_infedit.py
```

For further information about the input parameters, please refer to the [Handbook](https://github.com/sled-group/InfEdit/tree/website).

## Citation

```bibtex
@article{xu2023infedit,
  title={Inversion-Free Image Editing with Natural Language}, 
  author={Sihan Xu and Yidong Huang and Jiayi Pan and Ziqiao Ma and Joyce Chai},
  booktitle={Conference on Computer Vision and Pattern Recognition 2024},
  year={2024}
}
```

## Acknowledgements

We thank the awesome research works [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt), [LCM](https://huggingface.co/docs/diffusers/api/pipelines/latent_consistency_models) and [Masactrl](https://github.com/TencentARC/MasaCtrl).
