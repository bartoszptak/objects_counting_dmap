"""Looper implementation."""
from typing import Optional, List

import torch
import numpy as np
import matplotlib


class Looper():
    """Looper handles epoch loops, logging, and plotting."""

    def __init__(self,
                 network: torch.nn.Module,
                 device: torch.device,
                 loss: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_size: int,
                 plots: Optional[matplotlib.axes.Axes]=None,
                 validation: bool=False,
                 sliced: bool=False,
                 separated: bool=False):
        """
        Initialize Looper.

        Args:
            network: already initialized model
            device: a device model is working on
            loss: the cost function
            optimizer: already initialized optimizer link to network parameters
            data_loader: already initialized data loader
            dataset_size: no. of samples in dataset
            plot: matplotlib axes
            validation: flag to set train or eval mode

        """
        self.network = network
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.loader = data_loader
        self.size = dataset_size
        self.validation = validation
        self.sliced = sliced
        self.plots = plots
        self.running_loss = []
        self.separated = separated

    def run(self):
        """Run a single epoch loop.

        Returns:
            Mean absolute error.
        """
        # reset current results and add next entry for running loss
        self.true_values = []
        self.predicted_values = []
        self.running_loss.append(0)

        # set a proper mode: train or eval
        self.network.train(not self.validation)

        for image, gt in self.loader:
            # move images and labels to given device
            image = image.to(self.device)
            gt = gt.to(self.device)

            # clear accumulated gradient if in train mode
            if not self.validation:
                self.optimizer.zero_grad()

            # get model prediction (a density map)
            if self.separated:
                out = self.network(image[:,:3], image[:,3:])
            else:
                out = self.network(image)
                        
            # calculate loss and update running loss
            loss = self.loss(out, gt)
            self.running_loss[-1] += image.shape[0] * loss.item() / self.size
            #print(loss, image.shape[0] * loss.item() / self.size)
            # update weights if in train mode
            if not self.validation:
                loss.backward()
                self.optimizer.step()

            if self.validation and self.sliced:
                label = torch.zeros(1, 1, 1080, 1920).to(self.device)
                result = torch.zeros(1, 1, 1080, 1920).to(self.device)

                size = 608
                padding = size-32
                xs, ys = [0,576,1152,1312], [0,472]


                b_label = gt.squeeze()
                b_result = out.squeeze()

                assert b_label.size() == b_result.size()
                assert label.size() == result.size()
                for i, y in enumerate(ys):
                    for k, x in enumerate(xs):
                        part_l = b_label[i*len(xs)+k]
                        part_b = b_result[i*len(xs)+k]

                        if k == 0 and i == 0:
                            label[0, 0, y:y+size, x:x+size] = part_l
                            result[0, 0, y:y+size, x:x+size] = part_b
                        elif k != 0 and i != 0:
                            label[0, 0, y:y+32, x:x+32] = (label[0, 0, y:y+32, x:x+32] + part_l[:32, :32])/2
                            label[0, 0, y+32:y+size, x+32:x+size] = part_l[32:, 32:]
                            result[0, 0, y:y+32, x:x+32] = (result[0, 0, y:y+32, x:x+32] + part_b[:32, :32])/2
                            result[0, 0, y+32:y+size, x+32:x+size] = part_b[32:, 32:]
                        elif i == 0:
                            label[0, 0, y:y+size, x:x+32] = (label[0, 0, y:y+size, x:x+32] + part_l[:, :32])/2
                            label[0, 0, y:y+size, x+32:x+size] = part_l[:, 32:]
                            result[0, 0, y:y+size, x:x+32] = (result[0, 0, y:y+size, x:x+32] + part_b[:, :32])/2
                            result[0, 0, y:y+size, x+32:x+size] = part_b[:, 32:]
                        elif k == 0:
                            label[0, 0, y:y+32, x:x+size] = (label[0, 0, y:y+32, x:x+size] + part_l[:32, :])/2
                            label[0, 0, y+32:y+size, x:x+size] = part_l[32:, :]
                            result[0, 0, y:y+32, x:x+size] = (result[0, 0, y:y+32, x:x+size] + part_b[:32, :])/2
                            result[0, 0, y+32:y+size, x:x+size] = part_b[32:, :]
            else:
                result = out
                label = gt
            # loop over batch samples
            for true, predicted in zip(label, result):
                # integrate a density map to get no. of objects
                # note: density maps were normalized to 100 * no. of objects
                #       to make network learn better
                true_counts = torch.sum(true).item() / 100
                predicted_counts = torch.sum(predicted).item() / 100

                # update current epoch results
                self.true_values.append(true_counts)
                self.predicted_values.append(predicted_counts)

        # calculate errors and standard deviation
        self.update_errors()

        # update live plot
        if self.plots is not None:
            self.plot()

        # print epoch summary
        self.log()

        return self.mean_abs_err

    def update_errors(self):
        """
        Calculate errors and standard deviation based on current
        true and predicted values.
        """
        #print(self.true_values, len(self.true_values))
        #print(self.predicted_values, len(self.predicted_values))
        self.err = np.array([true - predicted for true, predicted in
                    zip(self.true_values, self.predicted_values)])
        self.abs_err = [np.abs(error) for error in self.err]
        self.mean_err = np.sum(self.err) / self.err.shape[0]
        self.mean_abs_err = np.sum(self.abs_err) / self.err.shape[0]
        self.std = np.array(self.err).std()

    def plot(self):
        """Plot true vs predicted counts and loss."""
        # true vs predicted counts
        true_line = [[0, max(self.true_values)]] * 2  # y = x
        self.plots[0].cla()
        self.plots[0].set_title('Train' if not self.validation else 'Valid')
        self.plots[0].set_xlabel('True value')
        self.plots[0].set_ylabel('Predicted value')
        self.plots[0].plot(*true_line, 'r-')
        self.plots[0].scatter(self.true_values, self.predicted_values)

        # loss
        epochs = np.arange(1, len(self.running_loss) + 1)
        self.plots[1].cla()
        self.plots[1].set_title('Train' if not self.validation else 'Valid')
        self.plots[1].set_xlabel('Epoch')
        self.plots[1].set_ylabel('Loss')
        self.plots[1].plot(epochs, self.running_loss)

        matplotlib.pyplot.pause(0.01)
        matplotlib.pyplot.tight_layout()

    def log(self):
        """Print current epoch results."""
        print(f"{'Train' if not self.validation else 'Valid'}:\n"
              f"\tAverage loss: {self.running_loss[-1]:3.4f}\n"
              f"\tMean error: {self.mean_err:3.3f}\n"
              f"\tMean absolute error: {self.mean_abs_err:3.3f}\n"
              f"\tError deviation: {self.std:3.3f}")
