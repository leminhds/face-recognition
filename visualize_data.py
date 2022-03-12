import utils

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = 'att_faces'

(X_train, y_train), (X_test, y_test) = utils.load_data(DATA_DIR)

subject_idx = 4
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3,figsize=(10,10))
subject_img_idx = np.where(y_train==subject_idx)[0].tolist()
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
    img = X_train[subject_img_idx[i]]
    img = img.reshape(img.shape[0], img.shape[1])
    ax.imshow(img, cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()