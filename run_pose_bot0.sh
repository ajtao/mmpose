export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=$PWD:../mmdetection:${PWD}/../vball_tracking


POSE_CFG=configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py
POSE_CKPT=https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth
BOTSORT_TAG=v5_12k_frames_just_tracks

MATCHES=(/mnt/g/data/vball/ball/skill_touches_ball/20*[0-3]_*)
MATCHES=( 20211004_lub_bel 20211002_ols_rze 20211001_luk_jas)
for _MATCH in "${MATCHES[@]}"
do
    MATCH=$(basename $_MATCH)
    CMD="python demo/top_down_video_demo_with_bot_trt.py $POSE_CFG --video-path /mnt/g/data/vball/squashed/polish_men/${MATCH}/end0.mp4 --output-root /mnt/f/output/mmpose/${BOTSORT_TAG} --match $MATCH --tracking-csv /mnt/g/output/heuristics/${BOTSORT_TAG}/${MATCH}/end0.csv"
    echo $CMD
    $CMD
done
