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
        if self.rank is not None:events=events[self.rank*16:(self.rank+1)*16]
        if not len(events):raise ValueError('Empty distributed batch')
        data=self.reader.data;first,_=image_plan(data['caption_image'],events,True)
        images=data['caption_image'][events[first]]
        features=np.stack([self.reader.get(i) for i in images]).astype(np.float32)
        seq=np.asarray(data['sequences'][events],np.int64);seq=seq[:,:int((seq!=0).sum(1).max())]
        return features,seq,events.copy()


def current_epoch_projection(state,step_seconds,valid_seconds):
    if state.get('epoch',1)==11:return 900
    steps=0 if state['phase']=='validate' else base.STEPS-state['batch_cursor']
    return 1.35*(steps*step_seconds+valid_seconds+90)+900
