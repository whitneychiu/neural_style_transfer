# Neural Style Transfer
This is the implementation of the neural style transfer [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780634)] The core idea is to take two images (a content image and a style image), and produce a new image that reflects the content of one but the artistic ”style” of the other. In this implementation, we use SqueezeNet (directly from torchvision) as our feature extractor.

### Dependencies
Refer to `environment.yaml`

### Repository Structure
- `style_transfer.py` contains the script to render style transfered images
```python
python style_transfer.py
```

### Results
![](https://github.com/whitneychiu/neural_style_transfer/blob/main/visualization/starry_me_cat.png?raw=true)

### Contact
This is my implementation of neural style transfer. If there are any questions, please contact **Whitney Chiu** <wchiu@gatech.edu>
