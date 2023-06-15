import torch
from torchviz import make_dot
from models import Generator, Discriminator

# Initialize the models
G = Generator()
D = Discriminator()

# Create dummy data to pass into the model
x = torch.randn(1, 3, 64, 64)
y = torch.randn(1, 3, 64, 64)

# Pass the data through the models
G_out = G(x)
D_out = D(x, y)

# Create the visualization
dot = make_dot(G_out, params=dict(list(G.named_parameters()) + list(D.named_parameters())))

# Display the graph
dot.view()
