import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from FCN_network import PatchGANDiscriminator
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F 

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(G, D, dataloader, optimizer_G, optimizer_D, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    G.train()
    D.train()
    running_loss_G = 0.0
    running_loss_D = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Zero the gradients
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        # 真实样本损失
        output_real = D(image_rgb, image_semantic)
        loss_real = F.binary_cross_entropy_with_logits(output_real, torch.ones_like(output_real))

        # Forward pass
        outputs = G(image_semantic)

        # 生成样本损失
        output_fake_detach = D(outputs.detach(), image_semantic)
        loss_fake_detach = F.binary_cross_entropy_with_logits(output_fake_detach, torch.zeros_like(output_fake_detach))

        # loss of D
        loss_D = (loss_real + loss_fake_detach) * 0.5

        # Backward pass and optimization of D
        if epoch % 4 ==0:
            loss_D.backward()
            optimizer_D.step()

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, outputs, 'train_results', epoch)

        # Compute the loss of G
        L1 = nn.L1Loss()
        output_fake = D(outputs, image_semantic)
        loss_G = F.binary_cross_entropy_with_logits(output_fake, torch.ones_like(output_fake)) * 0.3 + L1(outputs,image_semantic)

        # Backward pass and optimization
        loss_G.backward()
        optimizer_G.step()

        # Update running loss
        running_loss_D += loss_D.item()
        running_loss_G += loss_G.item()

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}')

def validate(model, dataloader, D, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_semantic)

            # Compute the loss
            output_fake = D(outputs, image_semantic)
            loss = F.binary_cross_entropy_with_logits(output_fake, torch.ones_like(output_fake))
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')


def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='edges2shoes_train_list.txt')
    val_dataset = FacadesDataset(list_file='edges2shoes_val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=350, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=350, shuffle=False, num_workers=8)

    # Initialize model, loss function, and optimizer
    G = FullyConvNetwork().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))

    D = PatchGANDiscriminator(3,3).to(device)
    optimizer_D = optim.Adam(D.parameters(), lr=1e-5, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler_G = StepLR(optimizer_G, step_size=200, gamma=0.2)
    scheduler_D = StepLR(optimizer_D, step_size=20, gamma=0.2)


    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        train_one_epoch(G, D, train_loader, optimizer_G, optimizer_D, device, epoch, num_epochs)
        validate(G, val_loader, D, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler_G.step()
        scheduler_D.step()

        # Save model checkpoint every 20 epochs
        if epoch % 5 == 0 :
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(G.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')
        # os.makedirs('checkpoints', exist_ok=True)
        # torch.save(G.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
