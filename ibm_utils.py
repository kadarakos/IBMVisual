import os
import json
from scipy.io import loadmat
from collections import defaultdict

class BasicDataProvider:
  def __init__(self, dataset):
    print 'Initializing data provider for dataset %s...' % (dataset, )

    # !assumptions on folder structure
    self.dataset_root = os.path.join('./datasets', dataset)
    self.image_root = os.path.join('./datasets', dataset, 'imgs')

    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, 'dataset.json')
    print 'BasicDataProvider: reading %s' % (dataset_path, )
    self.dataset = json.load(open(dataset_path, 'r'))

    # load the image features into memory
    features_path = os.path.join(self.dataset_root, 'vgg_feats.mat')
    print 'BasicDataProvider: reading %s' % (features_path, )
    features_struct = loadmat(features_path)
    self.features = features_struct['feats']

    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      self.split[img['split']].append(img)

  def _getImage(self, img):
    """ create an image structure for the driver """

    # lazily fill in some attributes
    if not 'local_file_path' in img: img['local_file_path'] = os.path.join(self.image_root, img['filename'])
    if not 'feat' in img: # also fill in the features
      feature_index = img['imgid'] # NOTE: imgid is an integer, and it indexes into features
      img['feat'] = self.features[:,feature_index]
    return img

  def _getSentence(self, sent):
    """ create a sentence structure for the driver """
    # NOOP for now
    return sent

  # PUBLIC FUNCTIONS

  def getSplitSize(self, split, ofwhat = 'sentences'):
    """ return size of a split, either number of sentences or number of images """
    if ofwhat == 'sentences': 
      return sum(len(img['sentences']) for img in self.split[split])
    else: # assume images
      return len(self.split[split])


  def iterImageSentencePair(self, split = 'train', max_images = -1):
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        yield out


  def iterSentences(self, split = 'train'):
    for img in self.split[split]: 
      for sent in img['sentences']:
        yield self._getSentence(sent)


def load_sets(split, dataset):
    """
    Utility to load the image features and captions for the
    training algorithm. It has the option to return the training.
    the validation or the testing slice of the dataset.
    It relies on the BasicDataProvider from https://github.com/karpathy/neuraltalk
    :param mode: train, test, valid or return the full set
    :return: rows is a list of captions, image feature matrix were every
             image feature-vector is there 5 times
    """
    """
    old version
    datareader = BasicDataProvider(dataset)
    rows = [x['tokens'] for x in datareader.iterSentences(split)]
    img_feats = [datareader.features.T[x['imgid']] for x in datareader.iterSentences(split)]
    """
    print "Loading captions"
    provider = BasicDataProvider(dataset)
    img_feats = []
    captions = []
    for i in provider.iterImageSentencePair(split):
      img_feats.append(i['image']['feat'])
      captions.append(i['sentence']['tokens'])
    print len(captions)
    print len(img_feats)
    return captions, img_feats