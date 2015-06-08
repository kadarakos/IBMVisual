# IBMVisual
Cross-situational word learning model, learning the meanings of words from image features.

## Usage
First you will need to download the image vectors from the image-caption
data sets and their corresponding captions. These are made available
on http://cs.stanford.edu/people/karpathy/deepimagesent/ and
detailed instructions can be found in the /dataset/ subfolder of this repo.

To create the visual word-vectors from the publication [coming up] 
from the Flickr8k data set you need to run 

```
python train.py vector online f8k train test
```


