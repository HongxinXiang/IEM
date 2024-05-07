import torch

from dataloader.dataset import get_supervised_singal_dim
from model.base.predictor import Predictor
from model.base.predictor import VisionPredictor


class ResumePredictors(torch.nn.Module):
    def __init__(self, resume, num_features=512, logger=None, **kwargs):
        super(ResumePredictors, self).__init__()
        self.log = print if logger is None else logger.info
        propertiesPredictor = Predictor(in_features=num_features,
                                        out_features=get_supervised_singal_dim()["basic_properties_info"])
        atomDistPredictor = Predictor(in_features=num_features,
                                      out_features=get_supervised_singal_dim()["atom_count_dict"])
        boundDistPredictor = Predictor(in_features=num_features,
                                       out_features=get_supervised_singal_dim()["bound_count_dict"])
        geometryPredictor = Predictor(in_features=num_features,
                                      out_features=get_supervised_singal_dim()["normalized_points_flatten"])

        self.log("===> Loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location='cpu')

        self.propertiesPredictor = self.load_resume_with_inconsistency(propertiesPredictor, checkpoint["propertiesPredictor"])
        self.atomDistPredictor = self.load_resume_with_inconsistency(atomDistPredictor, checkpoint["atomDistPredictor"])
        self.boundDistPredictor = self.load_resume_with_inconsistency(boundDistPredictor, checkpoint["boundDistPredictor"])
        self.geometryPredictor = self.load_resume_with_inconsistency(geometryPredictor, checkpoint["geometryPredictor"])
        self.log("load all the parameters of pre-trianed model.")

    def load_resume_with_inconsistency(self, model, ckpt_model_state_dict):
        ckp_keys = list(ckpt_model_state_dict)
        cur_keys = list(model.state_dict())
        len_ckp_keys = len(ckp_keys)
        len_cur_keys = len(cur_keys)
        model_sd = model.state_dict()
        for idx in range(min(len_ckp_keys, len_cur_keys)):
            ckp_key, cur_key = ckp_keys[idx], cur_keys[idx]
            model_sd[cur_key] = ckpt_model_state_dict[ckp_key]
        model.load_state_dict(model_sd)
        self.log("load the first {} parameters. layer number: model({}), pretrain({})".format(min(len_ckp_keys, len_cur_keys), len_cur_keys, len_ckp_keys))
        return model

    def forward(self, x):
        y1 = self.propertiesPredictor(x)
        y2 = self.atomDistPredictor(x)
        y3 = self.boundDistPredictor(x)
        y4 = self.geometryPredictor(x)
        return y1, y2, y3, y4


class ResumeSingleTeacher(torch.nn.Module):
    def __init__(self, model_name, resume_teacher, resume_teacher_name, logger=None, **kwargs):
        super(ResumeSingleTeacher, self).__init__()
        self.log = print if logger is None else logger.info
        self.model_name = model_name
        self.resume_teacher = resume_teacher
        self.resume_teacher_name = resume_teacher_name
        teacher = VisionPredictor(model_name=model_name, head_arch="none", num_tasks=None, pretrained=True)
        self.num_features = teacher.in_features

        self.log("===> Loading checkpoint '{}'".format(resume_teacher))
        checkpoint = torch.load(resume_teacher, map_location='cpu')

        self.teacher = self.load_resume_with_inconsistency(teacher, checkpoint[resume_teacher_name])
        self.log("load all the parameters of pre-trianed model.")

    def load_resume_with_inconsistency(self, model, ckpt_model_state_dict):
        ckp_keys = list(ckpt_model_state_dict)
        cur_keys = list(model.state_dict())
        len_ckp_keys = len(ckp_keys)
        len_cur_keys = len(cur_keys)
        model_sd = model.state_dict()
        for idx in range(min(len_ckp_keys, len_cur_keys)):
            ckp_key, cur_key = ckp_keys[idx], cur_keys[idx]
            model_sd[cur_key] = ckpt_model_state_dict[ckp_key]
        model.load_state_dict(model_sd)
        self.log("load the first {} parameters. layer number: model({}), pretrain({})".format(min(len_ckp_keys, len_cur_keys), len_cur_keys, len_ckp_keys))
        return model

    def forward(self, x):
        feat = self.teacher(x)
        return feat

