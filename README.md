# Robust nets
Pytorch  implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

Along with additions form [Sorting out Lipschitz function approximation](https://arxiv.org/abs/1811.05381) to ensure the network is lipschitz continuous.

This implementation is primarily designed to be easy to read and simple to modify.

## Results
Working on it

## Acknowledgements

- Significant amounts of code are borrowed from the [keras retinanet implementation](https://github.com/fizyr/keras-retinanet)
- The NMS module used is from the [pytorch faster-rcnn implementation](https://github.com/ruotianluo/pytorch-faster-rcnn)
- The iplementations of Group Sort, and the Bjorck Convolution is from the [LNets](https://github.com/cemanil/LNets)
