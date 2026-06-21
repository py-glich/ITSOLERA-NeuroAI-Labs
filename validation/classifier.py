import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
    Adapted for PyTorch.
    Input shape: [B, 1, Channels, Time]
    """
    def __init__(self, num_channels=8, seq_len=500, num_classes=2, F1=8, D=2, F2=16, dropout_rate=0.5):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        
        # Block 1: Temporal Conv & Spatial Depthwise Conv
        # 1D Temporal Convolution (kernel size 64 fits well for 250Hz sampling rate)
        self.temporal_conv = nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise Spatial Convolution (constrained to mix across channels)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        
        self.pooling1 = nn.AveragePooling2D = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 2: Separable Convolution
        self.separable_conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            # Pointwise
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        )
        self.bn3 = nn.BatchNorm2d(F2)
        
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Classifier
        # Calculate flat dimension
        # seq_len starts at 500
        # After temporal_conv: 501
        # After pooling1 (1, 4): 501 // 4 = 125
        # After separable_conv: 125 + 1 (padding) = 126
        # After pooling2 (1, 8): 126 // 8 = 15
        dummy_input = torch.zeros(1, 1, num_channels, seq_len)
        with torch.no_grad():
            x = self.temporal_conv(dummy_input)
            x = self.bn1(x)
            x = self.depthwise_conv(x)
            x = self.bn2(x)
            x = F.elu(x)
            x = self.pooling1(x)
            x = self.dropout1(x)
            x = self.separable_conv(x)
            x = self.bn3(x)
            x = F.elu(x)
            x = self.pooling2(x)
            x = self.dropout2(x)
            flat_dim = x.numel()
            
        self.fc = nn.Linear(flat_dim, num_classes)

    def forward(self, x):
        # Input shape: [B, C, T]
        # Reshape to [B, 1, C, T]
        if x.ndim == 3:
            x = x.unsqueeze(1)
            
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


def train_classifier(model, X_train, y_train, epochs=25, batch_size=32, lr=0.005, device='cpu'):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    num_samples = X_train.size(0)
    
    for epoch in range(epochs):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        correct = 0
        
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices].to(device), y_train[indices].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(indices)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == batch_y).sum().item()
            
    return model


def evaluate_classifier(model, X_test, y_test, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = X_test.size(0)
    
    with torch.no_grad():
        X_test_device = X_test.to(device)
        y_test_device = y_test.to(device)
        outputs = model(X_test_device)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_test_device).sum().item()
        
    accuracy = correct / total
    return accuracy
