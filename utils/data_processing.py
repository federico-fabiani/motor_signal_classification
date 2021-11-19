import random
from neo.io.neomatlabio import NeoMatlabIO
from elephant.conversion import BinnedSpikeTrain
from tensorflow.keras.utils import to_categorical
import logging
import quantities
import numpy as np
from bisect import bisect


class DataWrapper:
    def __init__(self):
        self.blk = None

    def load(self, file_path):
        print('Loading bulk file...')
        r = NeoMatlabIO(filename=file_path)
        self.blk = r.read_block()
        print(f'Number of segment loaded: {len(self.blk.segments)}')

    def get_epochs(self, epoch, nbins=None, bin_length=0.04, filter=True):
        """The function return a list of windows, with the brain activity associated to a specific task/epoch and its
         respective labels"""
        print(f'Selecting epoch {epoch} from correct segments...')
        tasks = []
        targets = []
        duration = []
        discarded = []
        boundaries = []
        spikes_list = []
        trial_epochs = []
        for i, seg in enumerate(self.blk.segments):
            # Every run performed by the monkey is checked, and if it is not correct it is ignored
            if filter and seg.annotations['correct'] == 0:
                discarded.append(i)
                continue
            evt = seg.events[0]
            labels = [str(lab).strip().lower() for lab in evt.labels]

            limits = None
            # Given a run, we check if the epoch of interest is part of it and in case we get the limits for it
            if epoch.lower() == 'all':
                limits = (evt[0], evt[-1], evt)

            elif epoch.lower() in labels:
                idx = labels.index(epoch.lower())
                limits = (evt[idx], evt[idx + 1], epoch)

            else:
                logging.info(f'\t Segment {i}, epoch not present in this segment')
            spk = seg.spiketrains

            if limits is not None:
                # For every found task we return a window
                if nbins is not None:
                    sparse_array = BinnedSpikeTrain(spk, n_bins=nbins, t_start=limits[0], t_stop=limits[1])
                    tasks.append(sparse_array.to_array().astype('float32'))
                    trial_epochs.append(epoch)
                else:
                    spikes_list.append(spk)
                    boundaries.append(limits)
                targets.append(seg.annotations['obj'])
                duration.append(limits[1] - limits[0])
            if len(targets) != len(spikes_list):
                print(i)
        logging.info(f'{len(discarded)} discarded segments: {discarded}')
        print(f'{len(discarded)} filtered out; {len(targets)} left')
        duration = np.array(duration)
        mu_t = duration.mean()
        sigma_t = duration.std()
        logging.info(
            f'{len(duration)} kept segments, with average duration for epoch {epoch}: {mu_t}s and std: {sigma_t}s')

        if nbins is None:
            average_bins = round(mu_t / bin_length)
            logging.info(f'Number of bins having average duration {bin_length}s: {average_bins}')
            targets2 = []
            discarded = 0
            for i in range(len(targets)):
                try:
                    sparse_array = BinnedSpikeTrain(spikes_list[i], n_bins=average_bins, t_start=boundaries[i][0],
                                                    t_stop=boundaries[i][1])
                    tasks.append(sparse_array.to_array().astype('float32'))
                    targets2.append(targets[i])
                    if epoch.lower() == 'all':
                        epochs = []
                        evt = list(boundaries[i][2].times)
                        labels = boundaries[i][2].labels
                        step = duration[i]/average_bins*quantities.s
                        for bin_i in range(average_bins):
                            t = evt[0]+step*(bin_i+1)
                            last_state = bisect(evt, t) - 1
                            epochs.append(labels[last_state])

                        trial_epochs.append(epochs)
                    else:
                        trial_epochs.append(boundaries[i][2])
                except ValueError:
                    discarded += 1
                    continue
        print(f'{discarded} lost due to warning; {len(targets2)} left')
        return np.array(tasks), np.array(targets2), trial_epochs, duration

    def get_epoch_plus_noise(self, target_epoch='go', bin_size_ms=40, filter=True, crop=True):
        """ The function return a list of windows of measurements of the brain activity, each one associated to the epoch
         of the experiment that the monkey was performing """
        measurements = []
        epochs = []
        bin_size = bin_size_ms * quantities.millisecond
        min_dim = None
        for i, seg in enumerate(self.blk.segments):
            # Every run performed by the monkey is checked, and if it is not correct it is ignored
            if filter and seg.annotations['correct'] == 0:
                print(f'\tSegment {i} discarded')
                continue
            evt = seg.events[0]
            lab = [str(lab).strip().lower() for lab in evt.labels]
            target_idx = lab.index(target_epoch)
            spk = seg.spiketrains
            # Add the target window
            target_sparse_array = BinnedSpikeTrain(spk, bin_size=bin_size, t_start=evt[target_idx],
                                                   t_stop=evt[target_idx + 1])
            measurements.append(target_sparse_array.to_array().astype('float32'))
            epochs.append(target_epoch)
            dim = measurements[-1].shape[1]
            rnd_start = evt[0] + random.random() * (evt[target_idx] - evt[0] - bin_size * dim)
            rnd_sparse_array = BinnedSpikeTrain(spk, bin_size=bin_size, t_start=rnd_start,
                                                t_stop=rnd_start + bin_size * dim)
            measurements.append(rnd_sparse_array.to_array().astype('float32'))
            epochs.append('random')
            if min_dim is None or dim < min_dim:
                min_dim = dim
        if crop:
            cropped_list = []
            for elem in measurements:
                cropped_list.append(elem[:, :min_dim])
            measurements = np.array(cropped_list)
        return measurements, epochs


class ObjectSelector:
    """ObjectSelector class embed a dictionary associating single object to a common shape, and it is useful to make it
    flexible to select which objects or shapes to use as classes for the model. Each method return a list where every entry
    is the element or list of elements, composing a class. Besides, a list of names for those classes is provided"""

    def __init__(self):
        self.objects_dict = {
            "mixed": ["11", "12", "13", "14", "15", "16"],
            "rings": ["21", "22", "23", "24", "25", "26"],
            "cubes": ["31", "32", "33", "34", "35", "36"],
            "balls": ["41", "42", "43", "44", "45", "46"],
            "hCylinder": ["51", "52", "53", "54", "55", "56", "57", "58"],  # 57 and 58 are 2 different ways to grasp 56
            "boxes": ["61", "62", "63", "64", "65", "66"],
            "vCylinder": ["71", "72", "73", "74", "75", "76"],
            "special": ["81", "82", "83", "84", "85", "86"],
            "special2": ["91", "92", "93", "94", "95", "96"],  # not clear labels found on one dataset
            "precision": "0",
            "strength": "1"
        }

    def get_dict(self):
        return self.objects_dict

    def get_shapes(self, shapes, group_labels=False):
        result = []
        names = []
        if type(shapes) != list:
            print(f'Input element {shapes} casted to list')
            shapes = [shapes]
        for shape in shapes:
            if shape in self.objects_dict.keys():
                if group_labels:
                    result.append(self.objects_dict[shape])
                    names.append(shape)
                else:
                    for obj in self.objects_dict[shape]:
                        result.append(obj)
                        names.append(obj)
            else:
                print(f'{shape} not present in objects dict; try with {self.objects_dict.keys()}')
        return list(zip(result, names))

    def get_non_special(self, group_labels=False, include_mixed=False):
        result = []
        names = []
        excluded = ['mixed', 'special', 'special2', 'strength', 'precision']
        if include_mixed:
            excluded.pop(excluded.index('mixed'))
        for k in self.objects_dict.keys():
            if k not in excluded:
                if group_labels:
                    result.append(self.objects_dict[k])
                    names.append(k)
                else:
                    for obj in self.objects_dict[k]:
                        result.append(obj)
                        names.append(obj)
        return list(zip(result, names))

    def get_all(self, group_labels=False):
        result = []
        names = []
        for k in self.objects_dict.keys():
            if group_labels:
                result.append(self.objects_dict[k])
                names.append(k)
            else:
                for obj in self.objects_dict[k]:
                    result.append(obj)
                    names.append(obj)
        return list(zip(result, names))


def load_dataset(file_path, epoch, nbins=None, bin_length=0.04):
    """ Load the desired measurements, from file or from cache if available """
    file = file_path.split('/')[-1].split('.')[0]
    logging.info(f'Loading dataset at {file_path}\nSelecting epoch {epoch}')
    TEMP = 'D:\\Workspaces\\PycharmProjects\\motor_signal_classification\\classifier\\temp'
    try:
        my_x = np.load(f'{TEMP}/{file}_{epoch}_X.npy')
        my_y = np.load(f'{TEMP}/{file}_{epoch}_Y.npy')
        my_states = np.load(f'{TEMP}/{file}_{epoch}_states.npy')
        my_duration = np.load(f'{TEMP}/{file}_{epoch}_duration.npy')
        logging.info(f'Windows and objects loaded from cache;\n\tX - {my_x.shape}\n\tY - {my_y.shape}')

    except IOError:
        my_wrapper = DataWrapper()
        my_wrapper.load(file_path)
        my_x, my_y, my_states, my_duration = my_wrapper.get_epochs(epoch, nbins=nbins, bin_length=bin_length)
        np.save(f'{TEMP}/{file}_{epoch}_X.npy', my_x)
        np.save(f'{TEMP}/{file}_{epoch}_Y.npy', my_y)
        np.save(f'{TEMP}/{file}_{epoch}_states.npy', my_states)
        np.save(f'{TEMP}/{file}_{epoch}_duration.npy', my_duration)
        logging.info('Windows and objects loaded from records;\n')

    logging.info(f'Loaded {len(my_y)} records')
    return my_x, my_y, my_states, my_duration


def preprocess_dataset(my_x, my_y, labelled_classes, one_hot_encoder=None, norm_sets=False, shuffle=True):
    if labelled_classes is not None:
        # re-associate labels according to the desired rule, recall that each entry of labels array is the list of
        # objects composing that class
        new_y = []
        new_x = []
        for y_idx, old_y in enumerate(my_y):
            for (new_class_elements, new_class_label) in labelled_classes:
                if type(new_class_elements) != list:
                    new_class_elements = [new_class_elements]
                if str(old_y) in new_class_elements:
                    new_y.append(new_class_label)
                    new_x.append(my_x[y_idx])
                    break
        my_y = np.array(new_y)
        logging.info(f'{len(new_x)}/{len(my_x)} recordings kept belonging to {len(labelled_classes)} classes')
        my_x = np.array(new_x, dtype='float32')

    if norm_sets:
        logging.info('Dataset normalized')
        my_x = (my_x - my_x.min()) / (my_x.max() - my_x.min())
    if shuffle:
        logging.info('Dataset shuffled')
        rnd = np.random.permutation(len(my_y))
        my_x = my_x[rnd]
        my_y = my_y[rnd]
    if one_hot_encoder is not None:
        logging.info('Dataset encoded with one-hot labels')
        my_y = to_categorical(one_hot_encoder.fit_transform(my_y))

    return my_x, my_y


def split_sets(my_x, my_y, tr_split, val_split, normalize_classes=False, repetitions=1):
    """ Split the windows and objects lists into train, validation and test set """
    if normalize_classes:
        # count how many times each class is present in the dataset
        unique_y, n_repetition = np.unique(my_y, return_counts=True, axis=0)
        logging.info(f'De-biasing dataset: (unique_label, n_repetitions)\n\t{list(zip(unique_y, n_repetition))}')
        unique_y = unique_y.tolist()
        keep = min(n_repetition)
        for idx, old_y in enumerate(my_y):
            if n_repetition[unique_y.index(old_y)] > keep:
                my_y = np.delete(my_y, idx, 0)
                my_x = np.delete(my_x, idx, 0)

    # Checking for duplicates
    u, c = np.unique(my_x, return_counts=True, axis=0)
    dup = u[c > 1]
    if dup.shape[0] != 0:
        logging.info('WARNING: duplicates found!')

    tr_idx = round(len(my_y) * tr_split)
    val_idx = round(len(my_y) * (tr_split + val_split))

    my_x_train = []
    my_y_train = []
    my_x_val = []
    my_y_val = []
    my_x_test = my_x[val_idx:]
    my_y_test = my_y[val_idx:]
    my_x = my_x[:val_idx]
    my_y = my_y[:val_idx]

    for rep in range(repetitions):
        rnd = np.random.permutation(len(my_y))
        my_x = my_x[rnd]
        my_y = my_y[rnd]
        my_x_train.append(my_x[:tr_idx])
        my_y_train.append(my_y[:tr_idx])
        my_x_val.append(my_x[tr_idx:])
        my_y_val.append(my_y[tr_idx:])


    # logging.info(f'Train: {my_x_train.shape}, {my_y_train.shape}')
    # logging.info(f'Validation: {my_x_val.shape}, {my_y_val.shape}')
    # logging.info(f'Test: {my_x_test.shape}, {my_y_test.shape}')

    return (np.squeeze(my_x_train), np.squeeze(my_y_train)), \
           (np.squeeze(my_x_val), np.squeeze(my_y_val)), \
           (np.squeeze(my_x_test), np.squeeze(my_y_test)),


def expanding_window_preprocessing(my_x, my_y, my_states, one_hot_encoder):
    result_x = []
    result_y = []
    result_state = []
    (n_trials, n_channels, n_steps) = my_x.shape
    for trial in range(n_trials):
        for time_step in range(n_steps-1):
            padded_window = np.zeros((n_channels, n_steps-time_step-2))
            padded_window = np.hstack((padded_window, my_x[trial][:, :time_step+1]))
            result_x.append(padded_window.astype('float32'))
            result_y.append(my_y[trial])
            result_state.append(my_states[trial][time_step+1])

    result_state = to_categorical(one_hot_encoder.fit_transform(result_state))
    return np.array(result_x, dtype='float32'), np.array(result_y), result_state


if __name__ == '__main__':
    FILE = 'MRec40'  # MRec40, ZRec50 or ZRec50_Mini
    PATH = f'../data/Objects Task DL Project/{FILE}.neo.mat'

    wrapper = DataWrapper()
    wrapper.load(PATH)
    x, y, t = wrapper.get_epochs('all')
    # x, y = wrapper.get_epoch_plus_noise()
    print(y)
    # print(f'Mean duration: {np.mean(t)} - std: {np.std(t)}')
    # print(t)

    # os = ObjectSelector()
    # # classes, clNames = os.get_all(group_labels=False)
    # classes, clNames = os.get_shapes('rings', group_labels=True)
    # print(f'Number of classes: {len(classes)}')
    # for pair in zip(classes, clNames):
    #     print(pair)
