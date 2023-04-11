# Pix2Video Video Editing using Image Diffusion
Implementation of the paper "[Pix2Video: Video Editing using Image Diffusion](https://duyguceylan.github.io/pix2video.github.io/)" 

This method doesn't require `complex attention map copy/modification` or any kinds of `fine-tuning` compared to fate-zero/Tune-A-Video. We generate video with only pretrained text2img stable diffusion model. 

## Setup

### Requirements

```
pip install -r requirements.txt
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. To enable xformers, set `enable_xformers_memory_efficient_attention=True` (default).

### Weights

**[Stable Diffusion]** [Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from Hugging Face ([stable-diffusion-2 · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-depth)).

## Usage

### Inference

To fine-tune the text-to-image diffusion models for text-to-video generation, run this command:

```shell
accelerate launch train_pix2video.py --config="configs/pix2video/man-skiing.yaml"
```

## Results

<table class="center">
<tr>
  <td><img src="./assets/a jeep car is moving on the beach.gif"></td>      
  <td><img src="./assets/a jeep car is moving on the snow.gif"></td>
  <td><img src="./assets/a sports car is moving on the road.gif"></td>
</tr>
<tr>
  <td width=33% style="text-align:center;">"a jeep car is moving on the beach"</td>
  <td width=33% style="text-align:center;">"a jeep car is moving on the snow"</td>
  <td width=33% style="text-align:center;">"a sports car is moving on the road"</td>
</tr>
</table>

<table class="center">
<tr>
  <td><img src="./assets/Iron Man is skiing on the snow.gif"></td>      
  <td><img src="./assets/A man is surfing on the sea.gif"></td>
  <td><img src="./assets/A man wearing red is skiing on the snow.gif"></td>
</tr>
<tr>
  <td width=33% style="text-align:center;">"Iron Man is skiing on the snow"</td>
  <td width=33% style="text-align:center;">"A man is surfing on the sea"</td>
  <td width=33% style="text-align:center;">"A man wearing red is skiing on the snow"</td>
</tr>
</table>

<table class="center">
<tr>
  <td><img src="./assets/Bat man is surfing.gif"></td>      
  <td><img src="./assets/Iron man is surfing.gif"></td>
  <td><img src="./assets/Donald Trump is surfing.gif"></td>
</tr>
<tr>
  <td width=33% style="text-align:center;">"Bat man is surfing"</td>
  <td width=33% style="text-align:center;">"Iron man is surfing"</td>
  <td width=33% style="text-align:center;">"Donald Trump is surfing"</td>
</tr>
</table>

<table class="center">
<tr>
  <td><img src="./assets/a rabbit is eating an orange.gif"></td>      
  <td><img src="./assets/a rabbit is eating a pizza.gif"></td>
  <td><img src="./assets/a puppy is eating an orange.gif"></td>
</tr>
<tr>
  <td width=33% style="text-align:center;">"a rabbit is eating an orange"</td>
  <td width=33% style="text-align:center;">"a rabbit is eating a pizza"</td>
  <td width=33% style="text-align:center;">"a puppy is eating an orange"</td>
</tr>
</table>


## Citations

If you make use of the work, please cite the paper.
```bibtex
@article{ceylan2023pix2video,
  title={Pix2Video: Video Editing using Image Diffusion},
  author={Ceylan, Duygu and Huang, Chun-Hao Paul and Mitra, Niloy J},
  journal={arXiv preprint arXiv:2303.12688},
  year={2023}
}
}
```

## Shoutouts

- This code builds on [diffusers](https://github.com/huggingface/diffusers) and [Tune-A-Video](https://github.com/showlab/Tune-A-Video). Thanks for open-sourcing!

## Contact

If you have any questions or better implementation ways, feel free to open an issue or contact me through e-mail (wangfuyun@smail.nju.edu.cn)

