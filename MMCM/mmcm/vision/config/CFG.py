"""使用easydict配置适用于工程的全局参数字典
后面可以再拆分 线上、线下、各自业务的全局字典，在使用时指定
"""

from easydict import EasyDict as edct

from ...utils.color_util import PolledColor
from .model_cfg import ModelCfg

CFG = edct()
CFG.application = None

CFG.update(ModelCfg)
CFG.device = "cuda"
CFG.model_name = "StableDiffusion"  # 从model_cfg中找到对应模型的路径
CFG.pipe_name = "StableDiffusionPipeline"  # 与diffusers中的pipeline类同名字符串
CFG.scheduler = None  # 与diffusers中的Scheduler类同名字符串
CFG.requires_safety_checker = True  # 是否使用 safety_checker
CFG.template_name = "default"  # 用于将输入的参数字典转化成模型prompt输入
CFG.prompt_type = "default"  # prompt_type不影响代码运行，纯粹方便对prompt分类理解
CFG.target_width = 512
CFG.target_height = 512
CFG.num_inference_steps = 50
CFG.guidance_scale = 7.0
CFG.strength = 0.8
CFG.image_height_in_table = 240
CFG.try_num_util_succeed = 20  # 每个prompt任务图像生成尝试次数，不成功就放弃
CFG.seed = None  # predictor 预测时的随机数生成器种子
# 一些专门定义的分隔符

# 属性字符串中有该字符串时，会通过外积运算裂变成多个任务，其他属性一模一样
CFG.task_fission_sep = "|"

# 属性字符串中有该字符串时，会裂变成多个任务描述
# 如`eyes`属性中有`small,black`，则真实文本会是`small eyes, black eyes`，任务数不会发生变化
# 该部分还没有真正起作用，需要后面看怎么真正参数化，目前如果需要该功能，请在表格中记住使用`,`作为分隔符。
CFG.atrtribute_fission_sep = ","

CFG.time_str_format = "%Y-%m-%d %H:%M:%S"

# 输出文本图像存储方式
# 是否给存的结果增加审核列，便于候选审核
CFG.add_checker_columns = False
CFG.validates = None

# video_map相关算法
CFG.SemanticsAlgo = "wenlan"
CFG.ClipTransitionAlgo = "TransnetV2"
CFG.SscneTransitionAlgo = "SceneSeg"
CFG.FaceAlgo = "insightface"
CFG.TrackAlgo = "deepsort"

# 剪辑相关配置
# 颜色可参考：http://www1.ynao.ac.cn/~jinhuahe/know_base/othertopics/computerissues/RGB_colortable.htm

# PRODUCTION, DEVELOP, DEBUG
CFG.RunningMode = "DEVELOP"
CFG.TalkingDetectionTh = -0.2
CFG.FocalHumanMaxid = 3
CFG.MoviepyVideoClipReadOffset = 0.2
CFG.RecallNum = 1
CFG.RankNum = 1

# MusicInfo
CFG.MSSLyricInterInterval = 4

# VideoInfo
CFG.VideoStart = 0.15
CFG.VideoEnd = 0.8

# VisualizationParameters
CFG.Font = "Courier"
# CFG.Font = "STXinwei" # 相对秀气的字体，比较适合MV
CFG.LyricFontSize = 25
# LyricTitleFontSize = CFG.LyricFontSize * 1.1

# Debug
# 是否可视化clip中的帧信息，目前打开了会非常卡，默认关闭
CFG.VisFrame = False
# 音乐转场点边界块，可视化颜色、宽度
CFG.MusicTransitionEdgeColor = (51, 161, 201)  # 孔雀蓝
CFG.MusicTransitionEdgeSize = (20, 60)

CFG.DebugFontSize = 30
CFG.DebugTextStrokeWidth = 2
CFG.DebugTextColors = ["red", "orange"]
CFG.DebugTextColorsCls = PolledColor(CFG.DebugTextColors)
CFG.DebugFrameTextDuration = 0.5
CFG.DebugFrameTextColor = "red"
