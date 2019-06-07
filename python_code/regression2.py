import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.fc2.weight)

   
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class LinearLayer(nn.Module):
    def __init__(self, input_size):
        super(LinearLayer, self).__init__()
        self.fc1 = nn.Linear(input_size,1)
        nn.init.xavier_uniform_(self.fc1.weight)
   
    def forward(self, x):
        out = self.fc1(x)
        return out
      
  

#cpu/gpu/seeds/cudnn
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float64)
seed = np.random.randint(1,100);

torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu',)

# data
input_size = 1
num_datapoints = 15
#x_train = np.random.normal(size=(num_datapoints,input_size))
#beta = 100*np.random.normal(size=input_size)
#c = 100*np.random.normal(size=1)
#y_train = np.dot(x_train,beta) + c


x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float64)

y_train = 2*x_train




#simulation parameters
neural_network = 1
learning_rate_GD = 0.001
learning_rate_GD_ODE = 1e-3
ratio_lr = round(learning_rate_GD/learning_rate_GD_ODE)
num_iterations_GD_ODE = 500
num_iterations_GD = int(num_iterations_GD_ODE/ratio_lr)
hidden_size = 10


#RUN 1:GD
torch.manual_seed(seed)
if neural_network:
    model = NeuralNet(input_size, hidden_size).to(device)
else:
    model = nn.Linear(1, 1)#LinearLayer(input_size).to(device)
    
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_GD)

#saving parameter and loss history  
loss_history_GD = np.zeros(num_iterations_GD)
params = torch.cat([param.data.view(-1) for param in model.parameters()]).clone()
num_params = len(params.data.numpy())
param_history_GD = np.zeros((num_params, num_iterations_GD))

for iteration in range(num_iterations_GD):
        inputs = torch.from_numpy(x_train)
        targets = torch.from_numpy(y_train)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss_history_GD[iteration] = loss.item()
        params = torch.cat([param.data.view(-1) for param in model.parameters()]).clone()
        param_history_GD[:,iteration]=params.data.numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (iteration) % 1000 == 0:
            print ('Iteration [{}/{}], Loss: {:.4f}'.format(iteration+1, num_iterations_GD, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')