# Demo for training Fashion MNIST autoencoder locally

Instructions

In order to get GPU acceleration, you need to use a Mac with an M-series chip. 
The code program might still be able to run on CPU if you have a different chip, but it might be slower.

0. Install [uv](https://docs.astral.sh/uv/)
  - If you get aren't able to install using the curl command on this page, you can install with `pip install uv`.
1. `uv venv`
2. `source .venv/bin/activate` 
3. If you're on a Mac with an M2 chip, you can use built-in GPU acceleration by running `time uv run train.py`. If you have a Windows/Linux laptop, you can use `time uv run train_pytorch.py`.
4. Now, open the directory using `open .` or similar. You will see two output images `reconstruction_comparison.png` and `pca_encoded.png`.
5. Experiment with different tags.
  - `-e` will change the number of epochs to train over
  - `-b` will change the batch size
  - `-l` will change the learning rate
  - `-r` will change the representation size

Training should just take about a minute.

Questions:
1. How does batch size affect runtime?
2. How does batch size training stability?
3. Can you modify the code to plot the loss after each gradient descent step?
4. Can you have the code run using a different [optimizer](https://ml-explore.github.io/mlx/build/html/python/optimizers.html#)? What is an optimizer? How does your choice of optimizer affect training performance and training time?

