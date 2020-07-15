from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

data=np.zeros((5,256,256,1))
for i in range(5):
    img = Image.open("my_dataset/{}.jpg".format(i+1))
    new_img = img.resize((256,256))
    new_img = new_img.convert("1")
    new_img = np.array(new_img).reshape(256,256,1)
    data[i] = new_img
np.save("alpha.npy",data)
