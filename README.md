# Visual Relation Detection in Images


This is the code of my master thesis **Visual Relation Detection in Images**. The project is based on PyTorch version of [MSDN](https://github.com/yikang-li/MSDN) and [faster R-CNN](https://github.com/longcw/faster_rcnn_pytorch).

I am still working on the project.

## Training
	Training FullNet

	```
	CUDA_VISIBLE_DEVICES=0 python train_net.py --load_RPN --saved_model_path=./output/RPN/RPN_region_full_best.h5 --enable_clip_gradient --step_size=3 --pool_type spatial_attention --spatial_type dual_mask --iteration_type use_brnn
	```

## Acknowledgement

We thank [yikang-li](https://github.com/yikang-li) and [longcw](https://github.com/longcw) for their generously releasing of their projects

## License:

The pre-trained models and the MSDN technique are released for uncommercial use.

Contact me if you have questions.
