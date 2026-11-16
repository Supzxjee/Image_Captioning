"""Full TRAIN image-grouped order and unique-image loader; NumPy only."""
import numpy as np
import full_data as base
from csa_sharing_data import image_plan


def grouped_order(caption_image, train_events, train_images, epoch):
    images=np.asarray(caption_image[:train_events])
    if type(epoch)!=int or not 1<=epoch<=10 or len(images)!=train_events or (np.diff(images)<0).any() or not np.array_equal(np.unique(images),np.arange(train_images)):
        raise ValueError('Invalid epoch or incomplete/sorted TRAIN caption mapping')
    boundaries=np.r_[0,np.flatnonzero(np.diff(images))+1,len(images)]
    return np.concatenate([np.arange(boundaries[i],boundaries[i+1],dtype=np.int64) for i in np.random.default_rng(42+epoch).permutation(train_images)])


def order_for_epoch(data,epoch):
    return grouped_order(data['caption_image'],base.TRAIN_EVENTS,base.TRAIN_IMAGES,epoch)


class SharedCaptionBatches(base.CaptionBatches):
    def __getitem__(self,item):
        if self.reader is None:self.reader=base.FeatureReader(self.root)
        start=(item+self.cursor)*self.batch_size;events=self.events[start:start+self.batch_size]
        if self.rank is not None:
            local_size=self.batch_size//2
            events=events[self.rank*local_size:(self.rank+1)*local_size]
        if not len(events):raise ValueError('Empty distributed batch')
        data=self.reader.data;first,_=image_plan(data['caption_image'],events,True)
        images=data['caption_image'][events[first]]
        features=np.stack([self.reader.get(i) for i in images]).astype(np.float32)
        seq=np.asarray(data['sequences'][events],np.int64);seq=seq[:,:int((seq!=0).sum(1).max())]
        return features,seq,events.copy()


class HybridCaptionBatches(SharedCaptionBatches):
    def __getitem__(self, item):
        if self.rank not in (0, 1) or self.batch_size != 32:
            raise ValueError('Hybrid sharing requires two training ranks and global batch 32')
        if (item + self.cursor + 1) * self.batch_size > len(self.events):
            return base.CaptionBatches.__getitem__(self, item)
        return super().__getitem__(item)


def current_epoch_projection(state,step_seconds,valid_seconds):
    if state.get('epoch',1)==11:return 900
    steps=0 if state['phase']=='validate' else base.STEPS-state['batch_cursor']
    return 1.35*(steps*step_seconds+valid_seconds+90)+900
