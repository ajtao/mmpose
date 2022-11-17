export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=$PWD:../mmdetection:${PWD}/../vball_tracking


POSE_CFG=configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py
POSE_CKPT=https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth
BOTSORT_TAG=12k_frames

MATCHES=(/mnt/g/data/vball/skill/20*[0-3]_*)

for _MATCH in "${MATCHES[@]}"
do
    MATCH=$(basename $_MATCH)
    CMD="python demo/top_down_video_demo_with_bot.py $POSE_CFG $POSE_CKPT --video-path /mnt/g/data/vball/squashed/squashed/${MATCH}/end0.mp4 --out-video-root vis_results/${BOTSORT_TAG} --tracking-csv /mnt/f/output/heuristics/${BOTSORT_TAG}/${MATCH}/end0.csv"
    echo $CMD
    $CMD
done
