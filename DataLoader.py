import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import time
from numpy import asarray
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np
from line_profiler import profile

def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab





    



class VideoQADataset(Dataset):

    def __init__(self, answers, ans_candidates, questions, app_feature_h5, video_ids,
                 app_feat_id_to_index):
        self.all_answers = answers
        self.all_questions = questions
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.app_feature_h5 = app_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.all_ans_candidates = ans_candidates

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = self.all_ans_candidates[index]
        question = self.all_questions[index]
        video_idx = self.all_video_ids[index].item()
        app_index = self.app_feat_id_to_index[str(video_idx)]
        question_text = clip.tokenize(question) 
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in ans_candidates])

        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['appearance_features'][app_index]  # (8, 16, 2048)

        appearance_feat = torch.from_numpy(appearance_feat)
        return (
            video_idx, answer, tokenized_prompts, appearance_feat, question_text,
        )

    def __len__(self):
        return len(self.all_questions)
    

# ########################################################Changed VideoQADataLoader for GPT########################################################
# # # # # # ############################################# Changed VideoQADataLoader for T5 ##############################################
class VideoQADataLoader(DataLoader):

    def __init__(self, **kwargs):
        annotation_file = kwargs.pop('annotation_file')
        annotation_file_qna = kwargs.pop('annotation_file_qna', None)

        # H5 파일에서 사용 가능한 비디오 ID 목록 가져오기
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        
        # ID를 인덱스에 매핑
        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        
        # 사용 가능한 비디오 ID 집합
        available_video_ids = set(str(id) for id in app_video_ids)
        
        # 첫 번째 어노테이션 파일 로드
        with open(annotation_file) as f:
            instances = f.readlines()
        
        # 헤더 제거
        _header = instances.pop(0)
        
        # 데이터 구조 초기화
        questions = []
        answers = []
        video_names = []
        video_ids = []
        ans_candidates = []
        
        # 첫 번째 어노테이션 파일 처리 (모두 유효하다고 가정)
        for instance in instances:
            data = json.loads(instance.strip())
            vid_id = data[1]
            video_ids.append(vid_id)
            vid_filename = data[2]
            video_names.append(vid_filename)
            q_body = data[4]
            questions.append(q_body)
            options = data[6:10]
            candidate = np.asarray([options[0], options[1], options[2], options[3]])
            ans_candidates.append(candidate)
            answer_idx = data[10]
            answers.append(answer_idx)
        
        print('number of questions from base annotation: %s' % len(questions))
        
        # 두 번째 어노테이션 파일 처리 (있는 경우)
        if annotation_file_qna:
            with open(annotation_file_qna) as f:
                instances_qna = f.readlines()
            
            # 헤더 제거 (필요한 경우)
            if len(instances_qna) > 0 and instances_qna[0].startswith('['):
                instances_qna.pop(0)
            
            qna_added = 0
            qna_skipped = 0
            
            for instance in instances_qna:
                data = json.loads(instance.strip())
                vid_id = data[1]
                
                # QnA 파일의 vid_id가 H5 파일에 있는지 확인
                if str(vid_id) in available_video_ids:
                    video_ids.append(vid_id)
                    vid_filename = data[2]
                    video_names.append(vid_filename)
                    q_body = data[4]
                    questions.append(q_body)
                    options = data[6:10]
                    candidate = np.asarray([options[0], options[1], options[2], options[3]])
                    ans_candidates.append(candidate)
                    answer_idx = data[10]
                    answers.append(answer_idx)
                    qna_added += 1
                else:
                    qna_skipped += 1
            
            print('number of questions added from QnA: %s' % qna_added)
            print('number of questions skipped from QnA: %s' % qna_skipped)
        
        print('total number of questions: %s' % len(questions))

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.dataset = VideoQADataset(
            answers, 
            ans_candidates, 
            questions, 
            self.app_feature_h5, 
            video_ids,
            app_feat_id_to_index
        )
       
        self.batch_size = kwargs['batch_size']

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
# # ###########################################Original VideoQADataLoader###########################################

# class VideoQADataLoader(DataLoader):

#     def __init__(self, **kwargs):
#         annotation_file = kwargs.pop('annotation_file')
#         ########## additional annotation file for qna
#         # annotation_file_qna = kwargs.pop('annotation_file_qna')

#         with open(annotation_file) as f:
#             instances = f.readlines()

#         # with open(annotation_file_qna) as f:
#         #     instances_qna = f.readlines()

#         _header = instances.pop(0)
#         questions = []
#         answers = []
#         video_names = []
#         video_ids = []
#         ans_candidates = []

        
#         for instance in instances:
#             data = json.loads(instance.strip())
#             vid_id = data[1]
#             video_ids.append(vid_id)
#             vid_filename = data[2]
#             video_names.append(vid_filename)
#             q_body = data[4]
#             questions.append(q_body)
#             options = data[6:10]
#             candidate = np.asarray( [ options[0], options[1], options[2], options[3] ] )
#             ans_candidates.append( candidate )
#             answer_idx = data[10]
#             answers.append(answer_idx)
#         print('number of questions: %s' % len(questions))

#         with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
#             app_video_ids = app_features_file['ids'][()]

#         app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}

#         self.app_feature_h5 = kwargs.pop('appearance_feat')
#         self.dataset = VideoQADataset(answers, ans_candidates, questions, self.app_feature_h5, video_ids,
#                                       app_feat_id_to_index, 
#                                       )
       
#         self.batch_size = kwargs['batch_size']

#         super().__init__(self.dataset, **kwargs)

#     def __len__(self):
#         return math.ceil(len(self.dataset) / self.batch_size)

# ###########################validation용###########################################
class VideoQAValidationDataset(Dataset):

    def __init__(self, answers, ans_candidates, questions, app_feature_h5, video_ids,
                 app_feat_id_to_index):
        self.all_answers = answers
        self.all_questions = questions
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.app_feature_h5 = app_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.all_ans_candidates = ans_candidates
        # self.q_types = q_types

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = self.all_ans_candidates[index]
        question = self.all_questions[index]
        video_idx = self.all_video_ids[index].item()
        app_index = self.app_feat_id_to_index[str(video_idx)]
        question_text = clip.tokenize(question) 
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in ans_candidates])
        # q_type = self.q_types[index]
        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['appearance_features'][app_index]  # (8, 16, 2048)

        appearance_feat = torch.from_numpy(appearance_feat)
        return (
            video_idx, answer, tokenized_prompts, appearance_feat, question_text
        )

    def __len__(self):
        return len(self.all_questions)


class VideoQAValidationDataLoader(DataLoader):

    def __init__(self, **kwargs):
        annotation_file = kwargs.pop('annotation_file')

        with open(annotation_file) as f:
            instances = f.readlines()
        _header = instances.pop(0)
        questions = []
        answers = []
        video_names = []
        video_ids = []
        ans_candidates = []
        # q_types = []

        for instance in instances:
            data = json.loads(instance.strip())
            vid_id = data[1]
            video_ids.append(vid_id)
            vid_filename = data[2]
            video_names.append(vid_filename)
            q_body = data[4]
            questions.append(q_body)
            options = data[6:10]
            candidate = np.asarray( [ options[0], options[1], options[2], options[3] ] )
            ans_candidates.append( candidate )
            answer_idx = data[10]
            answers.append(answer_idx)
            # q_type = data[5]
            # q_types.append(q_type)
        print('number of questions: %s' % len(questions))

        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]

        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.dataset = VideoQAValidationDataset(answers, ans_candidates, questions, self.app_feature_h5, video_ids,
                                      app_feat_id_to_index, 
                                    
                                      )
       
        self.batch_size = kwargs['batch_size']

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
####################################################################################




# class VideoQATestDataset(Dataset):
#     def __init__(self, answers, ans_candidates, questions, app_feature_h5, video_ids,
#                  app_feat_id_to_index):
#         self.all_answers = answers
#         self.all_questions = questions
#         self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
#         self.unique_video_ids = torch.unique(self.all_video_ids)
#         self.unique_video_ids = self.unique_video_ids.tolist()
#         self.video_ids = []
#         # for i, video_id in enumerate(self.unique_video_ids):
#         #     video_ids = self.all_video_ids[self.all_video_ids == video_id]
#         #     print(len(video_ids))
#         #     # for id in video_ids.tolist():
#         #         # print(self.all_video_ids[id])
#         #     self.video_ids.extend(video_ids.tolist())
#         # # print(self.video_ids)
#         # exit()
#         self.app_feature_h5 = app_feature_h5
#         self.app_feat_id_to_index = app_feat_id_to_index
#         self.all_ans_candidates = ans_candidates

#     def __getitem__(self, index):
#         answer = self.all_answers[index] if self.all_answers is not None else None
#         ans_candidates = self.all_ans_candidates[index]
#         question = self.all_questions[index]
#         video_idx = self.all_video_ids[index].item()
#         app_index = self.app_feat_id_to_index[str(video_idx)]
#         question_text = clip.tokenize(question) 
#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in ans_candidates])

#         with h5py.File(self.app_feature_h5, 'r') as f_app:
#             appearance_feat = f_app['appearance_features'][app_index]  # (8, 16, 2048)
#             # print(f_app['appearance_features'].shape)
#             # print(len(f_app['ids'][:]))
#             # print(len(f_app['ids'][index]))
#             # Print all available keys in the h5 file
#             # print("Available keys in h5 file:", list(f_app.keys()))
#             ########
#             # Get number of clips for specific video
#             # video_features = f_app['appearance_features'][app_index]
#             # num_clips = video_features.shape[0]
#             # if(num_clips != 8):
#             #     print(f"Number of clips for video {app_index}: {num_clips}")
#             ########

#             # exit()

#         appearance_feat = torch.from_numpy(appearance_feat)
#         return (
#             video_idx, answer, tokenized_prompts, appearance_feat, question_text,
#         )

#     def __len__(self):
#         return len(self.all_questions)


# class VideoQATestDataLoader(DataLoader):

#     def __init__(self, **kwargs):
#         annotation_file = kwargs.pop('annotation_file')

#         with open(annotation_file) as f:
#             instances = f.readlines()
#         _header = instances.pop(0)
#         questions = []
#         answers = []
#         video_names = []
#         video_ids = []
#         ans_candidates = []

#         for instance in instances:
#             data = json.loads(instance.strip())
#             vid_id = data[1]
#             video_ids.append(vid_id)
#             vid_filename = data[2]
#             video_names.append(vid_filename)
#             q_body = data[4]
#             questions.append(q_body)
#             options = data[6:10]
#             candidate = np.asarray( [ options[0], options[1], options[2], options[3] ] )
#             ans_candidates.append( candidate )
#             answer_idx = data[10]
#             answers.append(answer_idx)
#         print('number of questions: %s' % len(questions))

#         with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
#             app_video_ids = app_features_file['ids'][()]

#         app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}

#         self.app_feature_h5 = kwargs.pop('appearance_feat')
#         self.dataset = VideoQATestDataset(answers, ans_candidates, questions, self.app_feature_h5, video_ids,
#                                       app_feat_id_to_index, 
#                                       )
       
#         # self.batch_size = kwargs['batch_size']
#         self.batch_size = 8    # 클립 개수

#         super().__init__(self.dataset, **kwargs)

#     def __len__(self):
#         return math.ceil(len(self.dataset) / self.batch_size)


# ######################################Validation with 6 scores
# # ###########################validation용###########################################

class VideoQAValidationDataset_6Scores(Dataset):

    def __init__(self, answers, ans_candidates, questions, app_feature_h5, video_ids,
                 app_feat_id_to_index, q_types):
        self.all_answers = answers
        self.all_questions = questions
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.app_feature_h5 = app_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.all_ans_candidates = ans_candidates
        self.q_types = q_types

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = self.all_ans_candidates[index]
        question = self.all_questions[index]
        video_idx = self.all_video_ids[index].item()
        app_index = self.app_feat_id_to_index[str(video_idx)]
        question_text = clip.tokenize(question) 
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in ans_candidates])
        q_type = self.q_types[index]
        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['appearance_features'][app_index]  # (8, 16, 2048)

        appearance_feat = torch.from_numpy(appearance_feat)
        return (
            video_idx, answer, tokenized_prompts, appearance_feat, question_text, q_type
        )

    def __len__(self):
        return len(self.all_questions)


class VideoQAValidationDataLoader_6Scores(DataLoader):

    def __init__(self, **kwargs):
        annotation_file = kwargs.pop('annotation_file')

        with open(annotation_file) as f:
            instances = f.readlines()
        _header = instances.pop(0)
        questions = []
        answers = []
        video_names = []
        video_ids = []
        ans_candidates = []
        q_types = []

        for instance in instances:
            data = json.loads(instance.strip())
            vid_id = data[1]
            video_ids.append(vid_id)
            vid_filename = data[2]
            video_names.append(vid_filename)
            q_body = data[4]
            questions.append(q_body)
            options = data[6:10]
            candidate = np.asarray( [ options[0], options[1], options[2], options[3] ] )
            ans_candidates.append( candidate )
            answer_idx = data[10]
            answers.append(answer_idx)
            q_type = data[5]
            q_types.append(q_type)
        print('number of questions: %s' % len(questions))

        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]

        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.dataset = VideoQAValidationDataset_6Scores(answers, ans_candidates, questions, self.app_feature_h5, video_ids,
                                      app_feat_id_to_index, 
                                      q_types
                                      )
       
        self.batch_size = kwargs['batch_size']

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)