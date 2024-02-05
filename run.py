import os
import vcf
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import allel
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


input_dim = 123
import os

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        return list(map(int, f.read().strip().split()))

train_input_data = []  
directory_path = "train_vcf" 


for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        integers = read_txt_file(file_path)
        train_input_data.append(integers)

test_input_data = []  
directory_path = "test_vcf"  

for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        integers = read_txt_file(file_path)
        test_input_data.append(integers)

different_positions = 0
position_count = input_dim 

train_images = []
train_path = 'train_image'
for filename in os.listdir(train_path):
    if filename.endswith(".JPG"):
        image_path = os.path.join(train_path, filename)
        image = Image.open(image_path)
        train_images.append(np.array(image))

train_output_data = torch.tensor(train_images, dtype=torch.float32)

test_images = []
test_path = 'test_image'
for filename in os.listdir(test_path):
    if filename.endswith(".JPG"):
        image_path = os.path.join(test_path, filename)
        image = Image.open(image_path)
        test_images.append(np.array(image))

test_output_data = torch.tensor(test_images, dtype=torch.float32)
train_input_tensor = torch.tensor(train_input_data, dtype=torch.float32)

train_output_tensor = train_output_data.clone().detach()
test_input_tensor = torch.tensor(test_input_data, dtype=torch.float32)
test_output_tensor = test_output_data.clone().detach()
train_dataset = TensorDataset(train_input_tensor, train_output_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class ImprovedGenerator(nn.Module):
    def __init__(self, output_height, output_width):
        super(ImprovedGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, output_height * output_width)
        self.output_height = output_height
        self.output_width = output_width

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.output_height, self.output_width)
    

output_height, output_width = test_output_tensor.shape[1], test_output_tensor.shape[2]
model = ImprovedGenerator(output_height, output_width)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

min_loss = float('inf')

best_pred = None
best_target = None
num_epochs = 400

for epoch in tqdm(range(num_epochs)):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        if loss.item() < min_loss:
            min_loss = loss.item()
            best_pred = output.detach().cpu() 
            best_target = target.detach().cpu()

if best_pred is not None and best_target is not None:
    plt.figure()
    
    plt.subplot(1, 2, 1)
    plt.title('Best Predict')
    plt.imshow(best_pred[0].numpy(), cmap='gray')  
    
    plt.subplot(1, 2, 2)
    plt.title('Target')
    plt.imshow(best_target[0].numpy(), cmap='gray')  
    
    plt.savefig('BestPredict_vs._Target.png')




with torch.no_grad():
    test_generated_images = model(test_input_tensor)
    test_loss = criterion(test_generated_images, test_output_tensor)


for i in range(len(test_generated_images)):
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.title(f'Predict {i}')
    plt.imshow(test_generated_images[i].squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f'Target {i}')
    plt.imshow(test_output_tensor[i].squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.savefig(f'Predict_Target_{i}.JPG', bbox_inches='tight', pad_inches=0)
    plt.close()

print("Completed.")

