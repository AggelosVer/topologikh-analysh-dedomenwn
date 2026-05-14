from persim import PersistenceImager
import numpy as np

all_h0 = [np.array([[0.0, 1.0], [0.0, 5.0]])]
pimager = PersistenceImager(pixel_size=0.1)
pimager.fit(all_h0 + [np.array([[0.0, 0.0], [1.0, 1.0]])])
img = pimager.transform(all_h0)
print([i.shape for i in img])
