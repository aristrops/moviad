from typing import List, Tuple
import torch
import torch.nn.functional as F

from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor

class STFPM(torch.nn.Module):

    DEFAULT_PARAMETERS = {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.4,
        "weight_decay": 1e-4,
        "momentum": 0.9,
    }

    def __init__ (
        self,
        teacher:CustomFeatureExtractor,
        student:CustomFeatureExtractor,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def forward(self, batch: torch.Tensor):

        if self.training:
            teacher_features, student_features = None, None
            with torch.no_grad():
                teacher_features = self.teacher(batch)
            student_features = self.student(batch)

            return teacher_features, student_features

        else:
            student_features = self.student(batch)
            teacher_features = self.teacher(batch)

            return self.post_process(
                teacher_features, student_features, batch.shape[2:]
            )

    def __call__(self, batch: torch.Tensor):
        return self.forward(batch)

    def train(self, *args, **kwargs):
        self.teacher.model.eval()
        self.student.model.train()
        return super().train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.teacher.model.eval()
        self.student.model.eval()
        return super().eval(*args, **kwargs)


    def post_process(self, t_feat, s_feat, output_shape) -> torch.Tensor:

        """
        This method actually produces the anomaly maps for evalution purposes

        Args:
            - t_feat: teacher features maps
            - s_feat: student features maps

        Returns:
            - anomaly maps

        """

        device = t_feat[0].device
        score_maps = torch.tensor([1.0], device=device)
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(
                sm, size=output_shape, mode="bilinear", align_corners=False
            )
            # aggregate score map by element-wise product
            score_maps = score_maps * sm

        anomaly_scores = torch.max(score_maps.view(score_maps.size(0), -1), dim=1)[0]
        return score_maps, anomaly_scores