#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os
import sys
import logging
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    
    
    # initiate running totals
    running_corrects = 0
    running_loss = 0
    
    for inputs, labels in test_loader:
        # send data to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
            
        # update running correct predictions & loss
        running_corrects += torch.sum(preds == labels.data).item()
        running_loss += loss.item() * inputs.size(0)
    
    # calculate accuracy and loss percentages
    total_correct = running_corrects/len(test_loader)
    total_losses = running_loss/len(test_loader)
    
    logger.info(f"Test Accuracy: {running_corrects}/{len(test_loader)} = {100*total_correct}% , Test Loss: {running_loss}/{len(test_loader)} = {100*total_losses}%")


def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    best_loss = 1e6
    loss_counter = 0
    
    for e in range(epochs):
        logger.info(f"Epoch: {e}")
        
        for phase in ['train', 'valid']:
            if phase == train:
                model.train()
            else: 
                model.eval()
            running_loss = 0.0
            running_corrects = 0

        
            for inputs, labels in image_dataset[phase]:
                #send data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)            
                
                if phase == 'train':
                    #run backward pass and update weights  
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                #Make predictions
                _, preds = torch.max(outputs,1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # calculate accuracy and loss percentages
            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects// len(image_dataset[phase])
            
            # update loss_counter and best_loss if applicable 
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1
                    
            logger.info(f"Epoch {e}: Train Accuracy: {running_corrects}/{len(image_dataset[phase])} = {100*epoch_acc}% , Train Loss: {running_loss}/{len(image_dataset[phase])} = {100*epoch_loss}%, Best Loss: {best_loss: 0.4f}")        
            

    #check for early stopping
        if loss_counter == 1:
            break
    
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # use a pretrained resnet18 model with 18 layers
    model = models.resnet18(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(nn.Linear(num_features, 256),
                            nn.ReLU(inplace = True),
                            nn.Linear(256, 133)) #133 types of dogs
    
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    train_path = os.path.join(data, "train/")
    validation_path = os.path.join(data, "valid/")
    test_path = os.path.join(data, "test/")
    
    logger.info(f"Train_path: {train_path}, valid path: {validation_path}, test_path: {test_path}")
    
    training_transform = transforms.Compose([
    transforms.RandomResizedCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

    testing_transform = transforms.Compose([
    transforms.RandomResizedCrop((224,224)),
    transforms.ToTensor(),
    ])
    
    train_set = torchvision.datasets.ImageFolder(root = train_path, transform = training_transform)
    validation_set = torchvision.datasets.ImageFolder(root = validation_path, transform = testing_transform)
    test_set = torchvision.datasets.ImageFolder(root = test_path, transform = testing_transform)
    
    logger.info(f"Train_set size: {len(train_set)}, Valid_set size: {len(validation_set)}, test_set size: {len(test_set)}")
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True) 
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)
    
    logger.info(f"Train_loader size: {len(train_loader)}, Valid_loader size: {len(validation_loader)}, test_loader size: {len(test_loader)}")
    
    return train_loader, validation_loader, test_loader

def main(args):
    
    #logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}, Epochs: {args.epochs}')
    logger.info(f'Data Directory: {args.data_dir}')
    
    # check for GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device}")
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    # send model to GPU if available
    model = model.to(device)
    
    logger.info(f"Creating data loaders with batch size: {args.batch_size}")
    # create data loaders
    train_loader, validation_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)   
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr = args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("Start model training...")
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, args.epochs, device)
    logger.info("Model training complete")
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing model...")
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving model...")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument("--data_dir", type = str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument("-model_dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--output_dir", type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    parser.add_argument("--batch_size", type = int, default = 32, metavar = "N")
    parser.add_argument("--lr", type = float, default = 0.1, metavar = "LR")
    parser.add_argument("--epochs", type = int, default = 5)
    
    args=parser.parse_args()
    
    main(args)
