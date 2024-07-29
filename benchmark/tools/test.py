import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, obj_from_dict
# from mmcv.parallel import scatter, collate, MMDataParallel
from mmdet.parallel import scatter, collate, MMDataParallel
from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors

import os
import mmdet.flags as FLAGS


def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result,
                                     data_loader.dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--width_mult',
        default=1.0,
        type=float,
        help='Width mult')
    args = parser.parse_args()
    return args


def set_width_mult(m, width_mult):
     if hasattr(m, 'width_mult'):
         m.width_mult = width_mult
         if hasattr(m, 'onehot'):
             m.onehot[:, :, :, :] = 0.
             channels = m.num_features_list[
                 FLAGS.width_mult_list.index(m.width_mult)]
             m.onehot[:, :channels, :, :] = 1.
         elif hasattr(m, 'in_channels_list'):
             m.current_in_channels = m.in_channels_list[
                 FLAGS.width_mult_list.index(m.width_mult)]
             m.current_out_channels = m.out_channels_list[
                 FLAGS.width_mult_list.index(m.width_mult)]
             m.current_groups = m.groups_list[
                 FLAGS.width_mult_list.index(m.width_mult)]
         else:
             pass


def main():
    args = parse_args()
    FLAGS.width_mult_test_list = [args.width_mult]

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    if args.gpus == 1:
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint)
        model = MMDataParallel(model, device_ids=[0])
        for width_mult in FLAGS.width_mult_test_list:
            args.out = os.path.join('/tmp', 'results_{}.pkl'.format(width_mult))
            print('Start to inference at width: {}x'.format(width_mult))
            FLAGS.width_mult_current = width_mult
            model.apply(lambda m: set_width_mult(m, width_mult=width_mult))
            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                num_gpus=1,
                dist=False,
                shuffle=False)
            outputs = single_test(model, data_loader, args.show)
            if args.out:
                print('writing results to {}'.format(args.out))
                mmcv.dump(outputs, args.out)
                eval_types = args.eval
                if eval_types:
                    print('Starting evaluate {}'.format(' and '.join(eval_types)))
                    if eval_types == ['proposal_fast']:
                        result_file = args.out
                    else:
                        result_file = args.out + '.json'
                        results2json(dataset, outputs, result_file)
                    coco_eval(result_file, eval_types, dataset.coco)
    else:
        # model_args = cfg.model.copy()
        # model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        # model_type = getattr(detectors, model_args.pop('type'))
        # outputs = parallel_test(
        #     model_type,
        #     model_args,
        #     args.checkpoint,
        #     dataset,
        #     _data_func,
        #     range(args.gpus),
        #     workers_per_gpu=args.proc_per_gpu)
        raise NotImplementedError

    # if args.out:
        # print('writing results to {}'.format(args.out))
        # mmcv.dump(outputs, args.out)
        # eval_types = args.eval
        # if eval_types:
            # print('Starting evaluate {}'.format(' and '.join(eval_types)))
            # if eval_types == ['proposal_fast']:
                # result_file = args.out
            # else:
                # result_file = args.out + '.json'
                # results2json(dataset, outputs, result_file)
            # coco_eval(result_file, eval_types, dataset.coco)


if __name__ == '__main__':
    main()
