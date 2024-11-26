# DIP Assignment 3

In this assignment, two tasks are accomplished.

## 1. GAN

### 1.1 Run

To run an Encoder-Decoder network:
```bash
python train.py
```

To run GAN:
```bash
python GAN.py
```

### 1.2 Encoder-Decoder to GAN

To improve performance, a discriminator is introduced. The PatchDiscriminator (D) is used to judge whether the input image with its label is generated (fake) or real.

- **D** is trained with loss \(L_D\):

\[
L_D = L_{\text{R}} + L_{\text{F}}
\]

- **Loss_R**: Binary cross-entropy with logits between `output_R` and 1s.
- **Loss_F**: Binary cross-entropy with logits between `output_F` and 0s.

**The Generator (G)** is trained with loss \(L_G\):

\[
L_G = L_1 + \lambda \times L_D
\]

\(\lambda\) is introduced because **D** is too strong such that **G** cannot fool **D**.


### 1.3 Results

A pretrained model on the `edges2shoes` dataset:

```
path="\01_GAN\result\pix2pix.pth"
```

**Train Results:**

![Train Result 1](\01_GAN\result\train_res\result_1.png)

Other results are shown at: `\01_GAN\result\train_res\`

**Validation Results:**

![Validation Result 1](\01_GAN\result\val_res\result_1.png)

Other results are shown at: `\01_GAN\result\val_res\`

## 2. Auto-DragGAN

This is a combination of [DragGAN](https://github.com/XingangPan/DragGAN) and [face-alignment](https://github.com/1adrianb/face-alignment).

By auto-detecting the face and setting the point and target, one can realize deforming the image by one click.

### 2.1 Run

- To run DragGAN, please refer to [DragGAN](https://github.com/XingangPan/DragGAN).
- To run face-alignment, you can simply install it from pip:

  ```bash
  pip install face-alignment
  ```

It may be challenging to run the two projects successfully, so be patient.

- Download stylegan3-r-ffhqu-256x256.pkl or other human face pretrained GAN from (https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3)

### 2.2 Realization

We add two widgets to the GUI interface:

- **Close Mouth Button**
- **Laugh Button**

When you click the button, face-alignment will automatically detect the face landmarks of the image and show relative key points as source points. Deformed points relative to closing the mouth or laughing are target points.

Click start, and the image will deform to either shut up or laugh out loud.

### 2.3 Results

**Close Mouth GIF:**

![Close Mouth](02_DragGAN/res_demo/close.gif)

**Laugh GIF:**

![Laugh](02_DragGAN/res_demo/laugh.gif)
