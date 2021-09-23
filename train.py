import numpy as np
from tqdm import tqdm
import logging
import torch


def train(model, train_dataloader, validation_dataloader, optimizer, epoch, device, min_val_loss, lr_scheduler, early_stopping, n_time):
    # For recording time

    # For computing average of loss, metric, accuracy
    loss_list = []

    # Training
    model.train()
    
    for batch_idx, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(train_dataloader)):
        input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), labels.to(device)

        # Forward
        y_pred = model(input_ids, token_type_ids, attention_mask, labels)
            
        # Computing Loss
        loss = y_pred[0]

        # write history
        loss_list.append(loss.detach().cpu().numpy())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    # Validation
    model.eval()

    # For computing average of loss, metric, accuracy
    val_loss_list = []
    
    for val_batch_idx, (val_input_ids, val_token_type_ids, val_attention_mask, val_labels) in enumerate(tqdm(validation_dataloader)):
        val_input_ids, val_token_type_ids, val_attention_mask, val_labels = val_input_ids.to(device), val_token_type_ids.to(device), val_attention_mask.to(device), val_labels.to(device)

        # Forward
        val_y_pred = model(val_input_ids, val_token_type_ids, val_attention_mask, val_labels)
        
        # Computing Loss
        val_loss = val_y_pred[0]

        val_loss_list.append(val_loss.detach().cpu().numpy())

    avg_loss = sum(loss_list)/len(train_dataloader)
    val_avg_loss = sum(val_loss_list)/len(validation_dataloader)
    print(f"[Epoch {epoch}] Train Loss: {avg_loss}\nValidation Loss: {val_avg_loss}")

    if min_val_loss > val_avg_loss:
        torch.save(model, f'./checkpoints/{n_time}_Epoch{epoch}_val_loss{val_avg_loss:.3f}model.pt')

    if early_stopping is not None:
        early_stopping(val_avg_loss, model)

    return min(min_val_loss, val_avg_loss), avg_loss, val_avg_loss