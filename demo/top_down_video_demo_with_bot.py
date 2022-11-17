# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2

import mmcv

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, vis_pose_result)
from mmpose.datasets import DatasetInfo

import numpy as np

from vtrak.track_utils import read_tracking


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
        '--max-frames', type=int, default=12000)
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
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

    assert args.show or (args.out_video_root != '')

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
    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = video.fps
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        posecfg = os.path.splitext(os.path.basename(args.pose_config))[0]
        vidname = os.path.basename(os.path.dirname(args.video_path))
        vid_fn = os.path.join(args.out_video_root,
                              f'botsort_{vidname}_{posecfg}.mp4')
        csv_fn = os.path.join(args.out_video_root,
                              f'botsort_{vidname}_{posecfg}.csv')
        print(f'Writing to {vid_fn}')
        videoWriter = cv2.VideoWriter(
            vid_fn, fourcc,
            fps, size)
        csv_wr = open(csv_fn, 'w')

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

    print('Running inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):

        # MOT Frame numbering starts at 1
        fnum = frame_id + 1
        person_results = []
        tids = []
        if fnum in player_frames:
            # person_results = [{'bbox' [x1, y1, x2, y2, conf]}, ...]
            for tid, trk in player_frames[fnum].items():
                det = [trk.x, trk.y, trk.w, trk.h, 1.0]
                person_results.append({'bbox': np.array(det)})
                tids.append(trk.tid)
        else:
            print(f'didn\'t find {fnum} in player_frames')

        if len(person_results) > 12:
            print(f'WHOOPS, person_results length {len(person_results)}')
            breakpoint()

        if args.use_multi_frames:
            frames = collect_multi_frames(video, frame_id, indices,
                                          args.online)

        if not person_results:
            vis_frame = cur_frame
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
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

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

            # show the results
            vis_frame = vis_pose_result(
                pose_model,
                cur_frame,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=False)

        if args.show:
            cv2.imshow('Frame', vis_frame)

        if save_out_video:
            videoWriter.write(vis_frame)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_id >= args.max_frames:
            break

    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
