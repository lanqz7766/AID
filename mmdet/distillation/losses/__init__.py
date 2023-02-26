from .fgd import  FeatureLoss
from .gmask import Gmask_FeatureLoss
from .emask import Emask_FeatureLoss
from .cam_gmask import Cam_Gmask_FeatureLoss
from .cam_only import  Cam_only_FeatureLoss
from .cam_teacher import Teacher_Cam_FeatureLoss
from .cam_only_no_mask import Cam_only_no_gmask_FeatureLoss
__all__ = [
    'FeatureLoss', 'Gmask_FeatureLoss', 'Emask_FeatureLoss', 'Cam_Gmask_FeatureLoss', 'Cam_only_FeatureLoss',
    'Teacher_Cam_FeatureLoss', 'Cam_only_no_gmask_FeatureLoss'
]
