import os
import random
import numpy as np
import argparse
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from SpeechEnhancementDataset import SpeechDataset
from speech_enhancement_model import Model, ModelInferenceWrapper
import time


def train(model, train_loader, val_loader, epochs, max_samples, chunk_size, device):
    """
    Trains the speech enhancement model on chunked audio input.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        epochs (int): Number of training epochs.
        max_samples (int): Total number of samples per audio input (e.g., 1 second = 16000).
        chunk_size (int): Size of each chunk fed into the model at a time.
        device (str): Device to train on ("cuda" or "cpu").

    Returns:
        list: List of evaluation losses collected during training.
    """
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss() # More robust than MSE for audio
    evaluation_loss = []
    
    for epoch in range(epochs):    
        model.train()
        iteration = 0
        
        for batch in train_loader:
            noisy, clean = [x.to(device) for x in batch]
            start_time = time.time()
            states = None
            num_steps = int(max_samples/chunk_size) # Expected number of chunks per 1s audio
            
            # Process input in sequential chunks
            for i in range(num_steps):
                noisy_chunk = noisy[:,:,(i*chunk_size):(i+1)*chunk_size]
                output_chunk, states = model(noisy_chunk, states, device)
                
                if(i == 0):
                    output = output_chunk
                else:
                    output = torch.cat([output, output_chunk], dim=-1)
            
            # Compute loss between full predicted and clean waveform
            loss = criterion(output, clean)
            
            # Backpropagation and weight updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"[Iter {iteration}] Train Loss: {loss.item():.4f}")
            print(f"Time per Iteration: {time.time() - start_time}")
            
            # Periodic evaluation
            if (iteration) % EvaluationInterval == 0:
                start_time = time.time()
                eval_loss = evaluate(model, val_loader, max_samples, chunk_size, device)
                model.train() # Reset mode after evaluation
                evaluation_loss.append(eval_loss)
                print(f"Evaluation time: {time.time() - start_time}")
                
            iteration += 1

        # Save model checkpoint after each epoch
        torch.save(model.state_dict(), f"speech_enhancement_model_v2_epoch_{epoch+1}.pt")
        print(f"Saved model at epoch {epoch+1}")
    return evaluation_loss

def evaluate(model, val_loader, max_samples, chunk_size, device):
    """
    Evaluates the model on the validation set using L1 loss.

    Args:
        model (nn.Module): Trained speech enhancement model.
        val_loader (DataLoader): Dataloader for validation data.
        max_samples (int): Total number of samples per audio input.
        chunk_size (int): Size of each chunk to process at a time.
        device (str): Device to evaluate on.

    Returns:
        float: Average validation loss over the dataset.
    """
    
    model.eval()
    criterion = nn.L1Loss()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            noisy, clean = [x.to(device) for x in batch]
            states = None
            num_steps = int(max_samples/chunk_size)
            
            for i in range(num_steps):
                noisy_chunk = noisy[:,:,(i*chunk_size):(i+1)*chunk_size]
                output_chunk, states = model(noisy_chunk, states, device)
                
                if(i == 0):
                    output = output_chunk
                else:
                    output = torch.cat([output, output_chunk], dim=-1)
            
            total_loss += criterion(output, clean).item()
    
    print(f"[Eval] Avg Loss: {total_loss / len(val_loader):.4f}")
    
    return total_loss / len(val_loader)

def count_parameters(model):
    """
    Prints the number of total and trainable parameters in the model.

    Args:
        model (nn.Module): The neural network model.
    """
    
    Number_of_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Number_of_all_parameters = sum(p.numel() for p in model.parameters())     
    
    print("############### Model Information ###################")
    print(f"Total number of parameters: {Number_of_all_parameters}")
    print(f"Number of trainable parameters: {Number_of_trainable_parameters}")
    print("#####################################################")
 
    



if __name__ == "__main__":
    """
    Entry point for training the speech enhancement model.
    Loads dataset, initializes model, and runs training loop.
    """
    
    # --------------------------- Config ---------------------------
    data_set = "./DataSet/" # Training Data Set
    batch_size = 8 
    sampling_rate = 16000
    epochs = 5
    max_samples = 1 * sampling_rate # 1 second audio
    chunk_size = 64
    NumTrainingFiles = 4500
    NumValidationFiles = 100
    EvaluationInterval = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training Device: ", device)
    
    # --------------------------- Dataset Prep ---------------------------
    print("Loading and preparing data set ...")
    List_of_Files = os.listdir(data_set)
    random.shuffle(List_of_Files)
    Train_Files = List_of_Files[:NumTrainingFiles]
    Val_Files = List_of_Files[:NumValidationFiles] 

    train_dataset = SpeechDataset(data_set, Train_Files)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset = SpeechDataset(data_set, Val_Files)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # --------------------------- Model Setup ---------------------------
    print("Loading model ...")
    model = Model().to(device)
    count_parameters(model)
    
    # --------------------------- Training ------------------------------
    print("Training ...")
    train(model, train_loader, val_loader, epochs, max_samples, chunk_size, device)
    
    print("Training Complete")
    

