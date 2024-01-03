# Project-Deep-Learning
## GAN for Cancer Image Generation in Pathology
This README provides a step-by-step guide to reproduce the results of the provided GAN code for generating cancer images. The code uses PyTorch and torchvision for implementing a Generative Adversarial Network (GAN) with a Generator and Discriminator architecture.

Prerequisites

    Python 3.6 or higher
    PyTorch
    torchvision
    Pillow (PIL)

### Step 1: Set up the Environment
Make sure you have Python installed. Create a virtual environment (optional) and install the required packages:

    pip install torch torchvision Pillow
    
### Step 2: Dataset

Download [the dataset](https://drive.google.com/open?id=1LpgW85CVA48C8LnpmsDMdHqeCGHKsAxw) and place it in the appropriate directory. In our example, the dataset is assumed to be in the ".../images" directory. You can change the *root_dir* parameter in the CancerDataset class accordingly.

### Step 3: Run the Code

Run the provided code. Open a terminal and navigate to the directory containing the script. Run the following command:

    python GAN_in_Pathology.py
    
Or you can work on this project directly on Google Colaboratory by clicking on the file **".ipynb"**
    
### Step 4: Monitor Training Progress


The code will train the GAN for a specified number of epochs. You can monitor the training progress in the terminal. The generated images will be saved at regular intervals.


### Step 5: View Generated Images

After training, you can find the generated images in the specified directory. These images will be named with the format sample_epoch_batch.png.

## Customization
Feel free to customize the following parameters in the script according to your preferences:

    latent_dim: Dimensionality of the latent space.
    img_shape: Shape of the generated images.
    batch_size: Number of images in each batch during training.
    num_epochs: Number of training epochs.

Adjust these parameters based on your hardware capabilities and the size of your dataset for optimal results.

