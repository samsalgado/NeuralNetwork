import torch
import numpy as np
x = torch.randn(3, requires_grad=True)
print(x)
y=x+2
#Calculated Gradient using Backpropagation
print(y)
z = (4.2,21.9,20.7,21.1,19.2,19.4,19.4,24.5,21.3,17.8,17.9,14.1,14.6,14.6,23.7,23.5,21.5)
J = ()
print(z)

#X for set of Endpoints
#Y for set of prompts to remember
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
#Model Prediction
def forward(x):
    return w * x 
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

#Training
learning_rate = 0.01
n_iters = 100
#Training Loop
for epoch in range(n_iters):
    #Prediction = forward pass
    y_pred = forward(X)
    l = loss(Y, y_pred)
    #Gradient is backward pass
    l.backward()
    #Update Weights
    with torch.no_grad():
        w -= learning_rate * w.grad 
    w.grad.zero_()
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss ={l:.8f}')
print(f'Prediction after training: f(5) = {forward(5):.3f}')
    
