# U-NET-From-Scratch

This repository provides a clean, modular, and fully documented implementation of the U-Net convolutional neural network architecture, built from scratch using PyTorch. This project serves as an educational resource to demonstrate the core principles of U-Net, a foundational model for biomedical image segmentation

## What is U-Net?
The U-Net architecture is a powerful convolutional neural network designed for fast and precise semantic segmentation of images. Its key innovation is the symmetrical, U-shaped structure, which combines two paths:

- A contracting path (encoder) to capture context.

- An expansive path (decoder) to enable precise localization.

The network's most critical feature is the use of skip connections, which transfer feature maps from the contracting path to the expansive path. This allows the model to retain high-resolution spatial information lost during downsampling, leading to more accurate segmentation results.

## Key Features

* **Modular Design:** The U-Net architecture is broken down into reusable `nn.Module` components, including the `convolution_block`, `downsampling_block`, and `upsampling_block`.
* **Full U-Net Model:** The complete `UNET_Architecture` class connects all the components to form the characteristic U-shaped structure.
* **Skip Connections:** The implementation correctly incorporates skip connections, which are crucial for preserving fine-grained spatial details during upsampling.
* **Sanity Check:** The code includes a simple test to verify that the model's forward pass runs successfully with dummy data, confirming its structural integrity.
* **Binary Segmentation Output:** The model is configured with a final 1-channel output layer suitable for binary segmentation tasks (e.g., foreground/background).

## Contributing

This project is open for contributions. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.


## How to Use

### Prerequisites

To run the code, you need to have the following libraries installed:

```bash
pip install torch numpy
