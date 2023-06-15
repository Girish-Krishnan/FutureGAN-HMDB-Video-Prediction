import csv
import matplotlib.pyplot as plt

# Open the metrics CSV file
with open('metrics.csv', 'r') as f:
    reader = csv.reader(f)
    # Skip the header row
    next(reader)
    # Read the rest of the rows into lists
    epochs, inception_scores, fids = zip(*[(int(row[0]), float(row[1]), float(row[2])) for row in reader])

# Plot the Inception Score
plt.figure()
plt.plot(epochs, inception_scores)
plt.xlabel('Epoch')
plt.ylabel('Inception Score')
plt.title('Inception Score over Epochs')
plt.savefig('inception_score.png')

# Plot the FID
plt.figure()
plt.plot(epochs, fids)
plt.xlabel('Epoch')
plt.ylabel('FID')
plt.title('FID over Epochs')
plt.savefig('fid.png')

# Show the plots
plt.show()
