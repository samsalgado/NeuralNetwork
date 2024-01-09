import numpy as np
import torch 
weights = torch.ones(3, requires_grad=True)
#X for set of Endpoints
#Y for set of prompts to remember
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)
w = 0.0
#Model Prediction
def forward(x):
    return w * x 
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

def gradient(x,y, y_predicted):
    return np.dot(2*x, y_predicted -y)
print(f'Prediction before training: f(5) = {forward(5):.3f}')

#Training
learning_rate = 0.01
n_iters = 50
#Training Loop
for epoch in range(n_iters):
    #Prediction = forward pass
    y_pred = forward(X)
    l = loss(Y, y_pred)
    weight = gradient(X, Y, y_pred)
    w -= learning_rate * weight
    
    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss ={l:.8f}')
print(f'Prediction after training: f(5) = {forward(5):.3f}')