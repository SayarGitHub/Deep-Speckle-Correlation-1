import matplotlib.pyplot as plt
import numpy as np

from model import get_model_deep_speckle


model = get_model_deep_speckle()

model.load_weights('pretrained_weights.hdf5')

alpha = np.load("alpha.npy")

pred_alpha = model.predict(alpha, batch_size=2)

plt.figure()
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(alpha[i, :].squeeze(), cmap='hot')
    plt.axis('off')
    plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(pred_alpha[i, :, :, 0].squeeze(), cmap='gray')
    plt.axis('off')

plt.show()