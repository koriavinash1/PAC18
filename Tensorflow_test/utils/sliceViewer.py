import matplotlib.pyplot as plt
import numpy as np


class Viewer(object):
    currentAxis = 0
    currentChannel = 0
    currentVolume = None
    currentObject = None

    def remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'o':
            self.next_slice(ax)
        elif event.key == 'i':
            self.previous_slice(ax)
        elif event.key == 'u':
            self.cycle_axis(ax)
        elif event.key == 'k':
            self.cycle_channel(ax, inc=-1)
        elif event.key == 'l':
            self.cycle_channel(ax, inc=1)
        fig.canvas.draw()

    def setImage(self, ax, reshow=False):
        newFigure = None
        if self.currentAxis == 0:
            newFigure = self.currentVolume[ax.index, :, :]
        elif self.currentAxis == 1:
            newFigure = self.currentVolume[:, ax.index, :]
        elif self.currentAxis == 2:
            newFigure = np.transpose(self.currentVolume[:, :, ax.index])
        ax.images[0].set_array(newFigure)
        previousIndex = ax.index
        if reshow:
            plt.close('all')
            fig, ax = plt.subplots()
            ax.index = previousIndex
            ax.imshow(newFigure)
            cid = fig.canvas.mpl_connect('key_press_event', self.process_key)
            plt.show()

    def cycle_channel(self, ax, inc):
        if len(self.currentObject.shape) == 3:
            return
        self.currentChannel = min(max((self.currentChannel + inc), 0), self.currentObject.shape[3] - 1)
        self.currentVolume = self.currentObject[:,:,:,self.currentChannel]
        self.setImage(ax, reshow=True)

    def cycle_axis(self, ax):
        self.currentAxis = (self.currentAxis + 1) % 3
        self.setImage(ax)

    def previous_slice(self, ax):
        ax.index = max((ax.index - 1), 0)
        self.setImage(ax)

    def next_slice(self, ax):
        ax.index = min((ax.index + 1), self.currentVolume.shape[self.currentAxis] - 1)
        self.setImage(ax)

    def multi_slice_viewer(self, volume):
        self.remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots()
        self.currentObject = volume
        if len(volume.shape) == 3:
            self.currentVolume = volume
        else:
            self.currentVolume = image[:,:,:,0]
        ax.volume = self.currentVolume
        ax.index = self.currentVolume.shape[0] // 2
        ax.imshow(self.currentVolume[ax.index, :, :])
        cid = fig.canvas.mpl_connect('key_press_event', self.process_key)

imFile = '../aligned.npy'
# imFile = '../../dataVisualizations/maps/Reverse0/attention/AttentionMap0.npy'
image = np.load(imFile)
image  = np.squeeze(image)
v = Viewer()
v.multi_slice_viewer(image)

plt.show()
