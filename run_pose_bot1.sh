export CUDA_VISIBLE_DEVICES=1

PYTHONPATH=$PWD:../mmdetection:${PWD}/../vball_tracking


POSE_CFG=configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py
POSE_CKPT=https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth
BOTSORT_TAG=v5_12k_frames

# MATCHES=(/mnt/g/data/vball/skill/20*[4-9]_*)
MATCHES=(  20220908_poland_usa_left)
MATCHES=(/mnt/g/data/vball/ball/skill_touches_ball/20*[4-9]_*)

for _MATCH in "${MATCHES[@]}"
do
    MATCH=$(basename $_MATCH)
    CMD="python demo/top_down_video_demo_with_bot.py $POSE_CFG $POSE_CKPT --video-path /mnt/g/data/vball/squashed/squashed/${MATCH}/end0.mp4 --out-video-root vis_results/${BOTSORT_TAG} --tracking-csv /mnt/g/output/heuristics/${BOTSORT_TAG}/${MATCH}/end0.csv"
    echo $CMD
    $CMD
done
