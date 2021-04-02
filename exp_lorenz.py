import torch as th
from tqdm import tqdm
from lorenz import LorenzDataSet, LorenzPrior
from bnn import BNN
from bsde import BSDE


device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


def test(model, loader, n_samples=1):
    mse = 0
    with th.no_grad():
        for batch in loader:
            # batch[0] is an artefact as the loader returns a list of length 1
            batch = batch[0].to(device)
            for trajectory in batch:
                y_pred = model.predict(trajectory, n_samples)
                mse += th.mean((y_pred - trajectory[:,:model.dim])**2)
    return mse.item()/len(loader.dataset)


def train(model, loader, N, epochs=100, n_samples=1):
    for epoch in tqdm(range(epochs)):
        for batch in loader:
            batch = batch[0].to(device)
            model.step(batch, N, n_samples)






if __name__ == "__main__":
    # Train a model on the Lorenz Attractor Experiment
    len_trainseq = 24
    len_testseq = 24
    N_train = 20
    N_test = 20
    param_known = [False, False, False]
    # Load the data set or generate a new random data set if none exists so far
    lorenzdataset = LorenzDataSet(len_trainseq, len_testseq, N_train, N_test)

    bnn = BNN(n_hidden=100, n_in=lorenzdataset.n_dim + 1, n_out=lorenzdataset.n_dim).to(device)
    prior = LorenzPrior(param_known, prior_std=2.0).to(device)
    bsde = BSDE(drift_func=bnn, prior_process=prior, is_pac=True).to(device)

    train(bsde, lorenzdataset.train_loader, N_train, verbose=False, testing=False)
    print(f"{param_known}")
    print(f"Test(Train): {test(bsde, lorenzdataset.train_loader, 1)}")
    print(f"Test(Test): {test(bsde, lorenzdataset.test_loader, 1)}")


