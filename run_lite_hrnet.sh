export PYTHONPATH=${PWD}:../mmdetection

python demo/top_down_pose_tracking_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_30_coco_384x288.py \
    https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_coco_384x288-a3aef5c4_20210626.pth \
    --video-path /mnt/g/data/vball/squashed/squashed/20220908_poland_usa_left/end0.mp4 \
    --out-video-root vis_results

#     configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py \
#     https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth \

#    --video-path demo/resources/demo.mp4 \
