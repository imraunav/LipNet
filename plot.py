import pickle
import matplotlib.pyplot as plt

with open("lipnet_train_loss_gpu0.pkl", mode="rb") as file:
	loss = pickle.load(file)

plt.plot(loss)
plt.show()
