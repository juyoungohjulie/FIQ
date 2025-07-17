import argparse, os
import h5py
import skvideo.io
from PIL import Image
import sys
import torch
from torch import nn
import torchvision
import random
import numpy as np
from PIL import Image

from datautils import utils
from datautils import sutd_traffic
import clip
import decord
from tqdm import tqdm
import sys
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
# def _transform(n_px):
#     return Compose([
#         Resize((n_px, n_px), interpolation=BICUBIC),
#         # Resize(n_px, interpolation=BICUBIC),
#         # CenterCrop(n_px),
#         # _convert_image_to_rgb,
#         # ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])

def build_resnet():
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if not 'resnet' in args.model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    model.cuda()
    model.eval()
    return model


def build_resnext():
    model = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32,
                              sample_size=112, sample_duration=16,
                              last_fc=False)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    assert os.path.exists('preprocess/pretrained/resnext-101-kinetics.pth')
    model_data = torch.load('preprocess/pretrained/resnext-101-kinetics.pth', map_location='cpu')
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    return model

def build_resnet3d_slowfast(pretrained='pretrained/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth'): # slowfast
    model = SlowFastWrapper(pretrained='pretrained/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth')
    # model = ResNet3dSlowFast(pretrained='pretrained/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth')

    # model = nn.DataParallel(model, device_ids=None)
    model.eval()
    return model

def build_VideoMAEv2():
    model = extract_feature()
    return model


def run_batch(cur_batch, model, preprocess=None):
    """
    Args:
        cur_batch: treat a video as a batch of images
        model: ResNet model for feature extraction
    Returns:
        ResNet extracted feature.
    """


    if preprocess is not None:
        cur_batch = cur_batch.numpy()
        image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
        image_batch = torch.FloatTensor(image_batch).cuda()

        with torch.no_grad():
            image_batch = torch.autograd.Variable(image_batch)
        cur_batch = torch.tensor(cur_batch)
        
        if torch.cuda.is_available():
            image_batch = image_batch.cuda()
            model = model.cuda()


        image_batch = torch.stack([preprocess(img) for img in cur_batch]).cuda() 

        # print(image_batch.shape)

        # print(image_batch.mean(), image_batch.std())
    else:
        mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1)
        std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)
        image_batch = cur_batch.numpy().astype(np.float32)
        image_batch = (image_batch / 255.0 - mean) / std
        image_batch = torch.FloatTensor(image_batch).cuda()
        with torch.no_grad():
            image_batch = torch.autograd.Variable(image_batch)

        if torch.cuda.is_available():
            image_batch = image_batch.cuda()
            model = model.cuda()
    with torch.no_grad():
        feats = model.encode_image(image_batch).float() # this was for clip
    feats = feats.data.cpu().clone().numpy() # error, changed it to feats.cpu().clone().numpy()
    return feats


def extract_clips_with_uniform_frames(path, num_clips, num_frames_per_clip, target_fps=None):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
        target_fps: target fps of the video
    Returns:
        A list of raw features of clips.
    """
    valid = True
    clips = list()
    decord_vr = decord.VideoReader(path, num_threads=16)
    total_frames = len(decord_vr)

    clip_indices = np.linspace(0, total_frames - 1, num_frames_per_clip * num_clips, dtype=int)
    clip_indices = np.round(clip_indices).astype(int)
    decord.bridge.set_bridge('torch')
    clips_torch = decord_vr.get_batch(torch.tensor(clip_indices))
    # import ipdb; ipdb.set_trace()
    clips_torch = clips_torch.permute(0, 3, 1, 2) / 255.0
    for clip in torch.split(clips_torch, num_clips):
        clips.append(clip)
    # return clips_torch, valid # was clips originally
    return clips, valid
def extract_clips_with_consecutive_frames(path, num_clips, num_frames_per_clip, target_fps=None):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
        target_fps: target fps of the video
    Returns:
        A list of raw features of clips.
    """
    valid = True
    clips = list()
    decord_vr = decord.VideoReader(path, num_threads=16)

    original_fps = decord_vr.get_avg_fps()
    total_frames = len(decord_vr)
    max_frame_len = total_frames - 1
    if total_frames <= num_frames_per_clip:
        clip_indices = np.linspace(0, total_frames - 1, num_frames_per_clip, dtype=int)
        decord.bridge.set_bridge('torch')
        clips_torch = decord_vr.get_batch(torch.tensor(clip_indices))
        clips_torch = clips_torch.permute(0, 3, 1, 2) / 255.0
        clips = [clips_torch for i in range(num_clips)]
        return clips, valid
    else:
        if target_fps is not None:
            ratio = target_fps / original_fps
            total_frames_ = total_frames * ratio
            valid_length = (total_frames_ - num_frames_per_clip) > 0
            start_indices = np.linspace(0, total_frames_ - num_frames_per_clip, num_clips)
            end_indices = start_indices + num_frames_per_clip
            if not valid_length:
                clip_indices = np.linspace(0, total_frames - 1, num_frames_per_clip, dtype=int)
                decord.bridge.set_bridge('torch')
                clips_torch = decord_vr.get_batch(torch.tensor(clip_indices))
                clips_torch = clips_torch.permute(0, 3, 1, 2) / 255.0
                clips = [clips_torch for i in range(num_clips)]
                return clips, valid
            # print(start_indices, end_indices)

            extracting_indices = []
            for si, ei in zip(start_indices, end_indices):
                clip_indices = np.linspace(si, ei - 1, num_frames_per_clip)
                clip_indices = np.floor(clip_indices / ratio).astype(int)
                # clip_indices = np.clip(clip_indices, 0, max_frame_len) # To solve out of bound error 
                extracting_indices.append(clip_indices)
        else:
            start_indices = np.linspace(0, total_frames - num_frames_per_clip, num_clips, dtype=int)
            end_indices = start_indices + num_frames_per_clip

            extracting_indices = []
            for si, ei in zip(start_indices, end_indices):
                extracting_indices.append(np.arange(si, ei))
    
        decord.bridge.set_bridge('torch')
        decord_vr.get_batch(torch.tensor([extracting_indices[-1][-1]]))
        clips_torch = decord_vr.get_batch(torch.tensor(np.concatenate(extracting_indices)))
        clips_torch = clips_torch.permute(0, 3, 1, 2) / 255.0

        clips = []
        for i in range(num_clips):
            clip = clips_torch[i*num_frames_per_clip:(i+1)*num_frames_per_clip]
            clips.append(clip)

    return clips, valid

def generate_h5(model, video_ids, num_clips, outfile, device, preprocess=None, target_fps=25, uniform=False):
    """
    Args:
        model: loaded pretrained model for feature extraction
        video_ids: list of video ids
        num_clips: expected numbers of splitted clips
        outfile: path of output file to be written
    Returns:
        h5 file containing visual features of splitted clips.
    """
    if args.dataset == "tgif-qa":
        if not os.path.exists('data/tgif-qa/{}'.format(args.question_type)):
            os.makedirs('data/tgif-qa/{}'.format(args.question_type))
    else:
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))

    dataset_size = len(video_ids)

    for i, (video_path, video_id) in enumerate(video_ids):
        if not type(video_id) == int:
            print(video_id)
            print(video_path)

    with h5py.File(outfile, 'w') as fd:
        feat_dset = None
        feat_motion = None
        video_ids_dset = None
        i0 = 0
        _t = {'misc': utils.Timer()}
        for i, (video_path, video_id) in enumerate(tqdm(video_ids)):
            _t['misc'].tic()
            if uniform:
                clips, valid = extract_clips_with_uniform_frames(video_path, num_clips=num_clips, num_frames_per_clip=16, target_fps=target_fps)
            else:
                clips, valid = extract_clips_with_consecutive_frames(video_path, num_clips=num_clips, num_frames_per_clip=16, target_fps=target_fps)
            if args.feature_type == 'appearance':
                

                clip_feat = [];
                if valid:
                    # For uniform frames
                    for clip_id, clip in enumerate(clips):
                        feats = run_batch(clip, model, preprocess)  
                        clip_feat.append(feats)

                    # For consecutive frames
                    # clip_feat = run_batch(clips, model, preprocess)
                    # import ipdb; ipdb.set_trace()
                else:
                    clip_feat = np.zeros(shape=(num_clips, 16, 512))
                clip_feat = np.asarray(clip_feat)
                
                if feat_dset is None: 
                    C, F, D, = clip_feat.shape 
                    feat_dset = fd.create_dataset('appearance_features', (dataset_size, C, F, D),
                                                  dtype=np.float32)
                    feat_motion = fd.create_dataset('motion_features', (dataset_size, C, D),
                                                  dtype=np.float32)
                    video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=int) #dtype=np.int
            elif args.feature_type == 'motion':
                clip_torch = torch.FloatTensor(np.asarray(clips)).cuda()
                if valid:
                    clip_feat = model(clip_torch)
                    clip_feat = clip_feat.squeeze()
                    clip_feat = clip_feat.detach().cpu().numpy()
                else:
                    clip_feat = np.zeros(shape=(num_clips, 2048))
                if feat_dset is None:
                    C, D = clip_feat.shape
                    feat_dset = fd.create_dataset('resnext_features', (dataset_size, C, D),
                                                  dtype=np.float32)
                    video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=int) # dtype=np.int

            i1 = i0 + 1
            feat_dset[i0:i1] = clip_feat
            feat_motion[i0:i1] = clip_feat.mean(1) 
            video_ids_dset[i0:i1] = video_id
            i0 = i1
            _t['misc'].toc()
            if (i % 1000 == 0):
                print('{:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                      .format(i1, dataset_size, _t['misc'].average_time,
                              _t['misc'].average_time * (dataset_size - i1) / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='specify which gpu will be used')
    # dataset info
    parser.add_argument('--dataset', default='tgif-qa', choices=['tgif-qa', 'msvd-qa', 'msrvtt-qa', 'sutd-traffic'], type=str)
    parser.add_argument('--question_type', default='none', choices=['frameqa', 'count', 'transition', 'action', 'none'], type=str)
    # output
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default="data/{}/{}_{}_feat.h5", type=str)
    # image sizes
    parser.add_argument('--num_clips', default=8, type=int)
    parser.add_argument('--image_height', default=224, type=int)
    parser.add_argument('--image_width', default=224, type=int)

    # network params
    parser.add_argument('--model', default='resnet101', choices=['clip_image', 'clip_image_b16', 'clip_image_l14', 'resnet101', 'resnext101', 'resnet3d_slowfast', 'VideoMAE'], type=str)
    parser.add_argument('--seed', default='66666', type=int, help='random seed')
    parser.add_argument('--feature_type', default='appearance', type=str)
    parser.add_argument('--target_fps', default=25, type=int)
    parser.add_argument('--uniform', default=False, action='store_true')

    args = parser.parse_args()
    
    # set gpu
    if args.model != 'resnext101':
        torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # annotation files
    if args.dataset == 'tgif-qa':
        args.annotation_file = ''
        args.video_dir = ''
        args.outfile = 'data/{}/{}/{}_{}_{}_feat.h5'
        video_paths = tgif_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        generate_h5(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.question_type, args.dataset, args.question_type, args.feature_type))
    elif args.dataset == 'msrvtt-qa':
        args.annotation_file = ''
        args.video_dir = ''
        video_paths = msrvtt_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        generate_h5(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.dataset, args.feature_type))

    elif args.dataset == 'msvd-qa':
        args.annotation_file = ''
        args.video_dir = ''
        args.video_name_mapping = ''
        video_paths = msvd_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        generate_h5(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.dataset, args.feature_type))
        
    elif args.dataset == 'sutd-traffic':
        args.video_file = './data/annotation_file/R3_all.jsonl'
        args.video_dir = './data/raw_videos/'
        
        video_paths = sutd_traffic.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.model == 'clip_image':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
        elif args.model == 'clip_image_b16':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/16", device=device)
        elif args.model == 'clip_image_l14':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-L/14", device=device)
        elif args.model == 'resnet101':
            model = build_resnet()
            preprocess = None
        elif args.model == 'resnext101':
            model = build_resnext()
            preprocess = None
        elif args.model == 'resnet3d_slowfast':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_resnet3d_slowfast(pretrained='pretrained/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth') 
            preprocess = _transform(224)
        elif args.model == 'VideoMAE':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_VideoMAEv2(args)
            print("VideoMAE")
            # I3D, R2+1D, SlowFast, VideoMAE, InterVideo
        else:
            pass
        if args.uniform:
            args.outfile = f'./data/{args.dataset}/{args.dataset}_{args.feature_type}_feat_{args.model}_uniform.h5'
        else:
            args.outfile = f'./data/{args.dataset}/{args.dataset}_{args.feature_type}_feat_{args.model}_{args.target_fps}fps.h5'
        outfile = args.outfile
        generate_h5(model, video_paths, args.num_clips,
                    outfile, device, preprocess, args.target_fps, args.uniform)