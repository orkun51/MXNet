import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms

# Set the random seed for reproducibility
mx.random.seed(42)

# Define the transformation for the input data
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(train=True).transform_first(transformer)
test_dataset = datasets.MNIST(train=False).transform_first(transformer)

# Define the data loaders
batch_size = 64
train_loader = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = gluon.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model
net = nn.Sequential()
net.add(nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
net.add(nn.MaxPool2D(pool_size=2, strides=2))
net.add(nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
net.add(nn.MaxPool2D(pool_size=2, strides=2))
net.add(nn.Flatten())
net.add(nn.Dense(128, activation='relu'))
net.add(nn.Dense(10))

# Initialize the parameters of the model
net.initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())

# Define the loss function
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

# Define the optimizer
optimizer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    
    # Training
    for data, label in train_loader:
        with autograd.record():
            output = net(data)
            loss = loss_fn(output, label)
        loss.backward()
        optimizer.step(batch_size)
        
        train_loss += loss.mean().asscalar()
        train_acc += (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()
    
    # Validation
    test_loss = 0.0
    test_acc = 0.0
    
    for data, label in test_loader:
        output = net(data)
        loss = loss_fn(output, label)
        
        test_loss += loss.mean().asscalar()
        test_acc += (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()
    
    # Print the progress for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Loss: {train_loss / len(train_loader):.4f}, "
          f"Accuracy: {train_acc / len(train_loader):.4f}, "
          f"Val Loss: {test_loss / len(test_loader):.4f}, "
          f"Val Accuracy: {test_acc / len(test_loader):.4f}")
