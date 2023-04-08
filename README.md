# Pix2Video Video Editing using Image Diffusion
Implementation of the paper "Pix2Video: Video Editing using Image Diffusion" 

This method doesn't require `complex attention map copy/modification` or any kinds of `fine-tuning` compared to fate-zero/Tune-A-Video. We generate video with only pretrained text2img stable diffusion model. 

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
  <td width=33% style="text-align:center;">"a jeep car is moving on the snow"</td>
  <td width=33% style="text-align:center;">"a sports car is moving on the road"</td>
</tr>
</table>

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

## Citations

## Citation
If you make use of our work, please cite our paper.
```bibtex
@article{ceylan2023pix2video,
  title={Pix2Video: Video Editing using Image Diffusion},
  author={Ceylan, Duygu and Huang, Chun-Hao Paul and Mitra, Niloy J},
  journal={arXiv preprint arXiv:2303.12688},
  year={2023}
}
}