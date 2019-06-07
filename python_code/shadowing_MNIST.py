import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np



# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.relu = lambda x: x
        self.fc2 = nn.Linear(hidden_size, 1)  
        nn.init.xavier_uniform_(self.fc2.weight)

   
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
class LinearLayer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        #self.fc2 = nn.Linear(5, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        #nn.init.xavier_uniform_(self.fc2.weight)
   
    def forward(self, x):
        out = self.fc1(x)
        #out = self.fc2(out)
        return out
      
  

#cpu/gpu/seeds/cudnn
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loading data
train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(),download=True)

#getting data from Datalodear
batch_size = int(len(train_dataset))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

#transforming it to tensors
images, labels = next(iter(train_loader))
images = images.reshape(-1, 28*28).double().to(device)
labels = labels.to(device)

#network parameters
neural_network=0
input_size = 784
hidden_size = 5
num_classes = 2

#selecting just 2 digits
digit1 = 2
digit2 = 7
mask = (labels == digit1) | (labels == digit2)
images = images[mask]
labels = labels[mask]
images = images[1:10,:]
labels = labels[1:10]



labels[labels == digit1] = 0
labels[labels == digit2] = 1

#simulation parameters
learning_rate_GD = 1e-2
learning_rate_GD_ODE = 1e-3
ratio_lr = round(learning_rate_GD/learning_rate_GD_ODE)
num_iterations_GD_ODE = int(1e4)
num_iterations_GD = int(num_iterations_GD_ODE/ratio_lr)


for k in range(0,1):
    seed = np.random.randint(1,100);

    #RUN 1:GD
    torch.manual_seed(seed)
    if neural_network:
        model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    else:
        model = LinearLayer(input_size, num_classes).to(device)
    
    criterion = nn.MSELoss() #nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_GD)
    
    #saving parameter an loss history  
    loss_history_GD = np.zeros(num_iterations_GD)
    params = torch.cat([param.data.view(-1) for param in model.parameters()]).clone()
    num_params = len(params.data.numpy())
    param_history_GD = np.zeros((num_params, num_iterations_GD))
    
    for iteration in range(num_iterations_GD):
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels.double())
            loss_history_GD[iteration] = loss.item()
            params = torch.cat([param.view(-1) for param in model.parameters()])
            param_history_GD[:,iteration]=params.data.numpy()
            loss += 1*torch.mean(params**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (iteration) % 500 == 0:
                print ('Iteration [{}/{}], Loss: {:.4f}'.format(iteration+1, num_iterations_GD, loss.item()))
    
    
    #RUN 2:GD ODE
    torch.manual_seed(seed)
    if neural_network:
        model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    else:
        model = LinearLayer(input_size, num_classes).to(device)
    
    criterion = nn.MSELoss() #nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_GD_ODE)  
    loss_history_GD_ODE = np.zeros(num_iterations_GD_ODE)
    params = torch.cat([param.data.view(-1) for param in model.parameters()]).clone()
    num_params = len(params.data.numpy())
    param_history_GD_ODE = np.zeros((num_params, num_iterations_GD_ODE))
    
    for iteration in range(num_iterations_GD_ODE):
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels.double())
            loss_history_GD_ODE[iteration] = loss.item()
            params = torch.cat([param.view(-1) for param in model.parameters()])
            param_history_GD_ODE[:,iteration]=params.data.numpy()
            loss += 1*torch.mean(params**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            if (iteration) % 500 == 0:
                print ('Iteration [{}/{}], Loss: {:.4f}'.format(iteration+1, num_iterations_GD_ODE, loss.item()))
    
    
    #Getting the residuals to assess shadowing
    param_history_GD_ODE = param_history_GD_ODE[:,::ratio_lr]
    loss_history_GD_ODE = loss_history_GD_ODE[::ratio_lr]
    res =param_history_GD-param_history_GD_ODE
    
    
    #Plotting
    plt.figure(0)
    plt.plot(range(len(loss_history_GD_ODE)),loss_history_GD_ODE,'deepskyblue',range(len(loss_history_GD)),loss_history_GD,'orange')
    plt.show()
    plt.figure(1)
    plt.plot(range(len(loss_history_GD_ODE)),np.linalg.norm(res, axis=0))
    plt.xlabel('$k$',fontsize=20)
    plt.ylabel('$||x_k-y_k||$',fontsize=20)
    plt.show()



