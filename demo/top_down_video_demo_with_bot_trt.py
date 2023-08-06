# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2

import mmcv

from mmpose.apis import init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo
from mmdeploy_python import PoseDetector

import numpy as np
from tqdm import tqdm

from vtrak.track_utils import read_tracking
from vtrak.match_config import Match


def get_last_fnum(match, max_plays):
    mobj = Match(match, 'end0', use_offset=False,
                 max_plays=max_plays)
    last_second = mobj.plays[-1][-1]
    last_fnum = mobj.fps * (last_second + 1) + 1
    return last_fnum


def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Use player detector/tracking csv to localize humans
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument('--tracking-csv', type=str,
                        help='output from player detection/tracking pipeline')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--int8',
        action='store_true',
        default=False,
        help='whether to use int8 model')
    parser.add_argument(
        '--max-plays', type=int, default=None)
    parser.add_argument(
        '--max-frames', type=int, default=None)
    parser.add_argument(
        '--match', type=str, default=None)
    parser.add_argument(
        '--output-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--save-vid',
        action='store_true')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the pose'
        'estimation stage. Default: False.')

    args = parser.parse_args()

    if args.max_frames is not None:
        last_fnum = args.max_frames
    elif args.max_plays is not None:
        assert args.match is not None
        last_fnum = get_last_fnum(args.match, args.max_plays)
        print(f'Will run for {args.max_plays} plays, fnum {last_fnum}')
    else:
        last_fnum = None

    print('Initializing model...')
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # read video
    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Faild to load video file {args.video_path}'

    os.makedirs(args.output_root, exist_ok=True)
    posecfg = os.path.splitext(os.path.basename(args.pose_config))[0]
    vidname = os.path.basename(os.path.dirname(args.video_path))
    csv_fn = os.path.join(args.output_root,
                          f'botsort_{vidname}_{posecfg}.csv')
    csv_wr = open(csv_fn, 'w')

    if args.save_vid:
        fps = video.fps
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        vid_fn = os.path.join(args.output_root,
                              f'botsort_{vidname}_{posecfg}.mp4')
        print(f'Writing to {vid_fn}')
        videoWriter = cv2.VideoWriter(vid_fn, fourcc, fps, size)

    # frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert 'frame_indices_test' in pose_model.cfg.data.test.data_cfg
        indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    _, player_frames = read_tracking(args.tracking_csv)

    if args.int8:
        model_name = 'int8_dynamic-256x192'
    else:
        model_name = 'fp16_dynamic-256x192'
    trt_model_path = ('/home/atao/devel/public/mmdeploy/work-dirs/mmpose/'
                      f'topdown/hrnet/{model_name}')
    trt_detector = PoseDetector(model_path=trt_model_path, device_name='cuda',
                                device_id=0)

    print('Running inference...')
    pbar = tqdm(total=len(video), desc='pose estimation')
    for frame_id, cur_frame in enumerate(video):

        # MOT Frame numbering starts at 1
        fnum = frame_id + 1
        bboxes_xyxy = []
        tids = []
        if fnum in player_frames:
            for tid, trk in player_frames[fnum].items():
                tids.append(trk.tid)
                bbox = [trk.x, trk.y, trk.x+trk.w, trk.y+trk.h]
                bboxes_xyxy.append(bbox)
            pbar.set_postfix_str('pose estimation')
            pbar.update()
        else:
            pbar.set_postfix_str('skip, not in play')
            pbar.update()

        if len(bboxes_xyxy) > 12:
            print(f'WHOOPS, bboxes_xyxy length {len(bboxes_xyxy)}')

        if not bboxes_xyxy:
            trt_frame = cur_frame
        else:
            bboxes_batch = [bboxes_xyxy]
            imgs_batch = [cur_frame]
            trt_batch = trt_detector.batch(imgs=imgs_batch,
                                           bboxes=bboxes_batch)
            # pick first batch
            trt_results = trt_batch[0]
            # trt_results = (nplayers, 17, 3)

            pose_results = []
            for i in range(trt_results.shape[0]):
                bbox_with_score = bboxes_xyxy[i]
                bbox_with_score.append(1.0)
                result_person = {
                    'keypoints': trt_results[i],
                    'bbox': np.array(bbox_with_score),
                }
                pose_results.append(result_person)

            for idx in range(len(pose_results)):
                bbox = pose_results[idx]['bbox'].flatten()
                bbox = ','.join([str(x) for x in bbox])
                pose = pose_results[idx]['keypoints'].flatten()
                pose = ','.join([str(x) for x in pose])
                tid = tids[idx]
                line = f'{fnum},{tid},{bbox},{pose}\n'
                csv_wr.write(line)

            if args.save_vid:
                trt_frame = vis_pose_result(
                    pose_model,
                    cur_frame,
                    pose_results,
                    dataset=dataset,
                    dataset_info=dataset_info,
                    kpt_score_thr=args.kpt_thr,
                    radius=args.radius,
                    thickness=args.thickness,
                    show=False)

        if args.save_vid:
            videoWriter.write(trt_frame)

        if last_fnum is not None and fnum >= last_fnum:
            break

    if args.save_vid:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
