import torch
from .plot import show
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tqdm.notebook import tqdm



def train_network(generator, discriminator, criterion, 
                  optimizer_generator, optimizer_discriminator, 
                  n_epochs, dataloader, z_dim, inverse_transforms=None, 
                  epsilon=None, debug_run=False, show_images=True, label_smoothing = False,
                  save_path=None,
                  **kwargs):
    """
    Trains GAN by using batch-wise fake and real data inputs.
    
    params:
    --------------------
    generator: torch.Network
        Generator model to sample image from latent space.
        
    discriminator: torch.Network
        Discrimantor Model to classify into real or fake.
        
    criterion: 
        Cost-Function used for the network optimizatio
        
    optimizer: torch.Optimizer
        Optmizer for the network
        
    n_epochs: int
        Defines how many times the whole dateset should be fed through the network
        
    dataloader: torch.Dataloader 
        Dataloader with the batched dataset of real data.
        
    epsilon: float
        Stopping Criterion regarding to change in cost-function between two epochs.
        
    debug_run:
        If true than only one batch will be put through network.
        
    returns:
    ---------------------
    generator:
        Trained Torch Generator Model
        
    discriminator:
        Trained Torch Discriminator Model
        
    losses: dict
        dictionary of losses of all batches and Epochs.
        
    """
    print(20*'=', 'Start Training', 20*'=')
    # Init lists to keep track of loss
    training_loss_generator, training_loss_discriminator = [], []
    batch_loss = {'Dreal':[], 'Dfake':[], 'Gfake':[]}
    accuracy = {'real':[], 'fake':[], 'Gfake':[]}
    # Accuracy func
    calculate_accuracy = lambda y_true, y_pred: np.sum(y_true == y_pred) / y_true.shape[0]
    
    # Fixed noise for generating samples
    fixed_noise_dim = kwargs.get('fixed_noise_dim') if kwargs.get('fixed_noise_dim') else dataloader.batch_size
    fixed_noise = torch.randn(fixed_noise_dim, z_dim, 1, 1, device=device)
    
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Training on: {dev}')
    generator.to(dev), discriminator.to(dev)
    criterion.to(dev)

    generator.train(), discriminator.train()
    overall_length = len(dataloader)
    with tqdm(total=n_epochs*overall_length, disable=debug_run) as pbar:
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            discriminator_running_loss, generator_running_loss = 0.0, 0.0
            for i, data in enumerate(dataloader): 
                
                # Get Batch of real images and pass to Discriminator which first trains on only real data.
                real_images = data.to(dev)
                
                # Create Labels
                real_labels = torch.full(size=(real_images.shape[0], ), fill_value=real_label, 
                                         dtype=torch.float, device=device)
                fake_labels = torch.abs(torch.full(size=(real_images.shape[0], ), fill_value=fake_label, 
                                         dtype=torch.float, device=device))
                flipped_fake_labels = torch.full(size=(real_images.shape[0], ), fill_value=real_label, 
                                                 dtype=torch.float, device=device)  
                if label_smoothing:
                    real_labels += torch.normal(mean=0, std=.1, size=(real_labels.shape[0], ), device=dev)
                    fake_labels += torch.normal(mean=0, std=.1, size=(real_labels.shape[0], ), device=dev)
                    flipped_fake_labels += torch.normal(mean=0, std=.1, size=(real_labels.shape[0], ), device=dev)
                
                
                # Create noise from normal distribution
                # Generate Batch of latent vectors
                noise_vec = torch.randn(real_images.shape[0], Z_DIM, 1, 1, device=device)                
                
                # -----------------------------------------
                # REAL IMAGES - Training Discriminator 
                # -----------------------------------------
                discriminator.zero_grad()
                
                # Pass into Discriminator
                out = discriminator(real_images).view(-1)
                loss_real = criterion(out, real_labels)
                loss_real.backward()
                
                batch_loss['Dreal'].append(loss_real.mean().item())
                pred = torch.round(out)
                accuracy['real'].append(calculate_accuracy(real_labels.cpu().numpy(), 
                                                           pred.detach().cpu().numpy()))  
                 
                # -----------------------------------------
                # Fake Images - Discriminator
                # -----------------------------------------
                
                # Pass noise to generator
                fake_images = generator(noise_vec)
                # Passed generated images to discriminator
                out = discriminator(fake_images.detach()).view(-1)
                
                # Calculate Loss on Fake images
                loss_fake = criterion(out, fake_labels)
                loss_fake.backward()
                
                d_loss = loss_real + loss_fake
                
                # Udpate Discriminator Optimizer
                optimizer_discriminator.step()
                
                batch_loss['Dfake'].append(loss_fake.mean().item())
                
                # Calculate overall loss over both and append
                discriminator_running_loss += d_loss.mean().item()
                
                # Calculate accuracy on real images 
                pred = torch.round(out)
                accuracy['fake'].append(calculate_accuracy(fake_labels.cpu().numpy(), 
                                                           pred.detach().cpu().numpy()))
                # -----------------------------------------
                # FAKE - Training of Generator 
                # -----------------------------------------
                generator.zero_grad()
                                
                #fake_images = generator(G_z_vec)
                out = discriminator(fake_images).view(-1)
                loss_generator = criterion(out, flipped_fake_labels)
                loss_generator.backward()
                
                # Update G
                optimizer_generator.step()
                
                # Append Losses 
                batch_loss['Gfake'].append(loss_generator.mean().item())
                pred = torch.round(out)
                accuracy['Gfake'].append(calculate_accuracy(1-flipped_fake_labels.cpu().numpy(), 
                                                            pred.detach().cpu().numpy()))
                generator_running_loss += loss_generator.mean().item()
                
                # ----------------------------------------
                # calc and print stats
                pbar.set_description(f'Epoch: {epoch+1}/{n_epochs} // GRL: {round(generator_running_loss, 3)}  -- '+
                                     f'DRL: {round(discriminator_running_loss, 3)} ')
                pbar.update(1)
                if debug_run:
                    print('- Training Iteration passed. -')
                    break
                
            print(f'Epoch {epoch+1} // [GRL: {np.round(generator_running_loss, 3)}]  -- '+
                  f'[DRL: {np.round(discriminator_running_loss, 3)}] -- ' + 
                  f'[Accuracy (R/F): {round(np.mean(accuracy["real"][:-i]), 3)} - ',
                  f'{round(np.mean(accuracy["fake"][:-i]), 3)}]')
            
            training_loss_generator.append(generator_running_loss)
            training_loss_discriminator.append(discriminator_running_loss)
            
            if show_images:
                with torch.no_grad():
                    imgs = generator(fixed_noise).detach().cpu()
                    if inverse_transforms:
                        imgs = inverse_transforms(imgs)    
                    if save_path:
                        fig = show(make_grid(imgs), return_grid=True)
                        plt.savefig(f'{save_path}figure_{epoch+1}eps.png')
                        plt.show()
                    else:
                        show(make_grid(imgs), return_grid=False)
                    
            
            if epsilon:
                if epoch > 0:
                    diff = np.abs(training_loss_generator[-2] - generator_running_loss)
                    if diff < epsilon:
                        print('- Network Converged. Stopped Training. -')
                        break

            if debug_run:
                # Breaks loop 
                break
        
    print(20*'=', 'Finished Training', 20*'=')
    return generator, discriminator, dict(generator=training_loss_generator,
                                          discriminator=training_loss_discriminator,
                                          batch_loss=batch_loss, accuracy=accuracy)