# Description
This app takes an image, recognizes one or more horses in it by looking at pixels, and individually modifies the values of those pixels so that what comes out looks like a credible zebra. We won’t recognize anything zebra-like in the printout (or in the source code, for that matter): that’s because there’s nothing zebra-like in there. The network is a scaffold—the juice is in the weights.

# Usage
```
pip3 install -r requirements.txt
streamlit run main.py
```

![](demo.gif)

# Steps
## Deep learning Model
- Load the model architecture
- Load a pretrained weights and attach them to the model architecture (depends on the applications)

## Input Data
- import an input image(s)
- Perform preprocessing on the input image
- Create a batch of images to pass through inference

## Inference
- Set the model on eval mode
- Perform inference (eventually on GPU)


# Reference
- Deep Learning with Pytorch: Eli Stevens, Luca Antiga, Thomas Viehman