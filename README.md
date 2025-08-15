# U-NET-From-Scratch

This repository contains a complete, from-scratch implementation of the U-Net architecture for image segmentation using PyTorch. The goal of this project is to provide a clear, modular, and well-documented codebase that demonstrates the key principles of this foundational model in the computer vision field.

## Key Features

* **Modular Design:** The U-Net architecture is broken down into reusable `nn.Module` components, including the `convolution_block`, `downsampling_block`, and `upsampling_block`.
* **Full U-Net Model:** The complete `UNET_Architecture` class connects all the components to form the characteristic U-shaped structure.
* **Skip Connections:** The implementation correctly incorporates skip connections, which are crucial for preserving fine-grained spatial details during upsampling.
* **Sanity Check:** The code includes a simple test to verify that the model's forward pass runs successfully with dummy data, confirming its structural integrity.
* **Binary Segmentation Output:** The model is configured with a final 1-channel output layer suitable for binary segmentation tasks (e.g., foreground/background).

## How to Use

### Prerequisites

To run the code, you need to have the following libraries installed:

```bash
pip install torch numpy
