from neo.io.neomatlabio import NeoMatlabIO
from elephant.conversion import BinnedSpikeTrain
import quantities
import numpy as np


class DataWrapper:
    def __init__(self):
        self.blk = None

    def load(self, file_path):
        print('Loading bulk file...')
        r = NeoMatlabIO(filename=file_path)
        self.blk = r.read_block()
        print(f'Number of segment loaded: {len(self.blk.segments)}')

    def get_epochs(self, epoch, nbins=100):
        """The function return a list of windows, with the brain activity associated to a specific task/epoch and its
         respective labels"""
        print(f'Selecting epoch {epoch} from correct segments...')
        tasks = []
        targets = []
        duration = []
        for i, seg in enumerate(self.blk.segments):
            # Every run performed by the monkey is checked, and if it is not correct it is ignored
            if filter and seg.annotations['correct'] == 0:
                print(f'\tSegment {i} discarded')
                continue
            evt = seg.events[0]
            labels = [str(lab).strip().lower() for lab in evt.labels]

            limits = None
            # Given a run, we check if the epoch of interest is part of it and in case we get the limits for it
            if epoch.lower() in labels:
                idx = labels.index(epoch.lower())
                limits = (evt[idx], evt[idx + 1], epoch)
            else:
                print(f'\t Segment {i}, epoch not present in this segment')
            spk = seg.spiketrains

            if limits is not None:
                # For every found task we return a window
                sparse_array = BinnedSpikeTrain(spk, n_bins=nbins, t_start=limits[0], t_stop=limits[1])
                tasks.append(sparse_array.to_array().astype('float32'))
                targets.append(seg.annotations['obj'])
                duration.append(limits[1]-limits[0])
        return np.array(tasks), targets, duration


if __name__ == '__main__':
    FILE = 'ZRec50_Mini'  # MRec40, ZRec50 or ZRec50_Mini
    PATH = f'../data/Objects Task DL Project/{FILE}.neo.mat'

    wrapper = DataWrapper()
    wrapper.load(PATH)
    x, y, t = wrapper.get_epochs('Hold')
    print(f'Mean duration: {np.mean(t)} - std: {np.std(t)}')
    print(t)
