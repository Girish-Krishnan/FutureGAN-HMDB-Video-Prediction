import numpy as np
import matplotlib.pyplot as plt

# Load losses
lossesG = np.load('lossesG.npy')
lossesD = np.load('lossesD.npy')

# Plot losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(lossesG,label="G")
plt.plot(lossesD,label="D")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
