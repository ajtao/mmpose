# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2

import mmcv
import ffmpegcv

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, vis_pose_result)
from mmpose.datasets import DatasetInfo

import numpy as np
from tqdm import tqdm

from vtrak.track_utils import read_tracking
from vtrak.match_config import Match
from vtrak.vball_misc import safe_vid_rd


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
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument('--tracking-csv', type=str,
                        help='output from player detection/tracking pipeline')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--max-plays', type=int, default=None)
    parser.add_argument(
        '--max-frames', type=int, default=None)
    parser.add_argument(
        '--match', type=str, default=None)
    parser.add_argument(
        '--outdir',
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
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

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
    # video = mmcv.VideoReader(args.video_path)
    video = safe_vid_rd(args.video_path)

    os.makedirs(args.outdir, exist_ok=True)
    # Unified outputs:
    csv_fn = os.path.join(args.outdir, 'pose.csv')
    vid_fn = os.path.join(args.outdir, 'pose.mp4')
    csv_wr = open(csv_fn, 'w')

    if args.save_vid:
        fps = video.fps
        videoWriter = ffmpegcv.noblock(ffmpegcv.VideoWriterNV,
                                       vid_fn,
                                       codec='hevc',
                                       fps=fps)

    _, player_frames = read_tracking(args.tracking_csv)

    print('Running inference...')
    pbar = tqdm(total=len(video), desc='pose estimation', mininterval=10)
    for frame_id, cur_frame in enumerate(video):

        # MOT Frame numbering starts at 1
        fnum = frame_id + 1
        person_results = []
        tids = []
        if fnum in player_frames:
            for tid, trk in player_frames[fnum].items():
                tids.append(trk.tid)
                det = [trk.x, trk.y, trk.w, trk.h, 1.0]
                person_results.append({'bbox': np.array(det)})
            pbar.update()
        else:
            pbar.update()

        if len(person_results) > 12:
            print(f'WHOOPS, person_results length {len(person_results)}')

        if not person_results:
            trt_frame = cur_frame
        else:
            # test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                frames if args.use_multi_frames else cur_frame,
                person_results,
                bbox_thr=args.bbox_thr,
                format='xywh',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None)

            if len(pose_results) > 12:
                print(f'WHOOPS, pose_results length {len(pose_results)}')
                breakpoint()

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
