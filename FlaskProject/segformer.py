import colorsys
import copy
import time
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

# 从nets模块导入SegFormer
try:
    from nets.segformer import SegFormer
except ImportError:
    print("警告: 无法导入nets.segformer模块")
    class SegFormer:
        def __init__(self, *args, **kwargs):
            raise ImportError("无法导入SegFormer网络模型")

from utils.utils import cvtColor, preprocess_input, resize_image, show_config


# -----------------------------------------------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、backbone和num_classes都需要修改！
#   如果出现shape不匹配，一定要注意训练时的model_path、backbone和num_classes的修改
# -----------------------------------------------------------------------------------#
class SegFormer_Segmentation(object):
    _defaults = {
        # -------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        # -------------------------------------------------------------------#
        "model_path": "logs/best_epoch_weights.pth",
        # ----------------------------------------#
        #   所需要区分的类的个数+1
        # ----------------------------------------#
        "num_classes": 21,
        # ----------------------------------------#
        #   所使用的的主干网络：
        #   b0、b1、b2、b3、b4、b5
        # ----------------------------------------#
        "phi": "b2",
        # ----------------------------------------#
        #   输入图片的大小
        # ----------------------------------------#
        "input_shape": [768, 768],
        # -------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        # -------------------------------------------------#
        "mix_type": 0,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    # ---------------------------------------------------#
    #   初始化SegFormer
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        if self.num_classes <= 21:
            # 设置颜色：索引0为背景(0,0,0)，索引1为patches(128,0,0)红色，索引2为inclusion(0,128,0)绿色
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128),
                           (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # ---------------------------------------------------#
        #   获得模型
        # ---------------------------------------------------#
        self.generate()

        # 用于保存上次检测结果的类别统计信息
        self.last_classes_nums = None
        # 用于保存置信度信息
        self.last_confidence_info = {}

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        self.net = SegFormer(num_classes=self.num_classes, phi=self.phi, pretrained=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None, confidence_threshold=0.6, show_confidence=False,
                     mode="predict", img_name=""):
        # ---------------------------------------------------------#
        #   转换成RGB图像，防止灰度图在预测时报错
        # ---------------------------------------------------------#
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # ---------------------------------------------------------#
        #   Resize 图像，确保输入大小匹配
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   形态学膨胀 + 连通组件分析，合并相邻区域
            # ---------------------------------------------------#
            def post_process_segmentation(pr):
                """ 合并相邻缺陷区域，防止分割成多个部分 """
                binary_mask = (pr > 0).astype(np.uint8)  # 仅保留缺陷区域
                kernel = np.ones((3, 3), np.uint8)  # 使用 3x3 核，减少膨胀影响
                dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)  # 形态学膨胀

                # 连通组件分析
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated_mask)

                # 重新映射 pr，使相邻区域归为同一类别
                new_pr = np.zeros_like(pr)
                for i in range(1, num_labels):  # 跳过背景
                    mask = (labels == i)
                    dominant_class = np.argmax(np.bincount(pr[mask].flatten()))  # 找该区域中最多的类别
                    new_pr[mask] = dominant_class  # 统一类别

                return new_pr

            # ---------------------------------------------------#
            #   进行模型预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            # --------------------------------------#
            #   截取掉灰条部分
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            # ---------------------------------------------------#
            #   进行 Resize
            #   使用 INTER_NEAREST 避免插值导致分类错误
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            # ---------------------------------------------------#
            #   不使用置信度，只取最大概率类别
            # ---------------------------------------------------#
            if mode == "predict":
                pred_classes = np.argmax(pr, axis=-1)
                pred_classes = post_process_segmentation(pred_classes)
                
                # 保存类别统计信息
                if count and name_classes:
                    classes_nums = np.zeros([self.num_classes])
                    total_points_num = orininal_h * orininal_w
                    print('-' * 63)
                    print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
                    print('-' * 63)
                    for i in range(self.num_classes):
                        num = np.sum(pred_classes == i)
                        ratio = num / total_points_num * 100
                        if num > 0:
                            print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                            print('-' * 63)
                        classes_nums[i] = num
                    print("classes_nums:", classes_nums)
                    self.last_classes_nums = classes_nums

            # ---------------------------------------------------#
            #   其他模式使用置信度筛选
            # ---------------------------------------------------#
            elif mode == "confidence":
                confidences, pred_classes = np.max(pr, axis=-1), np.argmax(pr, axis=-1)
                pred_classes[confidences < float(confidence_threshold)] = 0
                
                # 保存置信度统计信息
                if count and name_classes:
                    classes_nums = np.zeros([self.num_classes])
                    confidence_info = {}
                    total_points_num = orininal_h * orininal_w
                    print('-' * 78)
                    print("|%25s | %15s | %15s | %15s|" % ("Key", "Value", "Ratio", "Confidence"))
                    print('-' * 78)
                    for i in range(self.num_classes):
                        mask = (pred_classes == i)
                        num = np.sum(mask)
                        ratio = num / total_points_num * 100
                        avg_confidence = np.mean(confidences[mask]) if num > 0 else 0
                        if num > 0:
                            print("|%25s | %15s | %14.2f%% | %14.2f|" % (
                                str(name_classes[i]), str(num), ratio, avg_confidence))
                            print('-' * 78)
                            # 保存置信度信息
                            confidence_info[name_classes[i]] = {
                                'count': int(num),
                                'ratio': float(ratio),
                                'avg_confidence': float(avg_confidence)
                            }
                        classes_nums[i] = num
                    print("classes_nums:", classes_nums)
                    self.last_classes_nums = classes_nums
                    self.last_confidence_info = confidence_info

            # ---------------------------------------------------#
            #   热力图模式
            # ---------------------------------------------------#
            elif mode == "heatmap":
                confidences, pred_classes = np.max(pr, axis=-1), np.argmax(pr, axis=-1)
                confidence_img = (confidences * 255).astype(np.uint8)
                confidence_img = cv2.applyColorMap(confidence_img, cv2.COLORMAP_JET)
                
                # 确保 old_img 和 confidence_img 都是 np.uint8 格式的 NumPy 数组
                if isinstance(old_img, Image.Image):
                    old_img = np.array(old_img)
                if isinstance(confidence_img, Image.Image):
                    confidence_img = np.array(confidence_img)

                old_img = old_img.astype(np.uint8)
                confidence_img = confidence_img.astype(np.uint8)

                # 使用加权叠加函数
                confidence_overlay = cv2.addWeighted(old_img, 0.6, confidence_img, 0.4, 0)

                # 保存热力图结果
                confidence_overlay = Image.fromarray(confidence_overlay)

                # 确保文件路径有有效的扩展名
                img_name = os.path.basename(img_name)  # 获取有效的文件名，避免路径问题

                # 如果文件名为空，使用默认名称
                if not img_name:
                    img_name = "unknown_image"

                # 构建输出路径，不包含 heatmap_ 前缀
                output_path = os.path.join("D:/aminuos/segformer-pytorch-master/test_results", f"{img_name}")

                # 添加扩展名，如果需要其他格式，可以修改这里
                if not output_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    output_path += '.jpg'  # 默认保存为 .jpg

                # 确保文件夹存在，如果文件夹不存在则创建
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 保存热力图结果到指定目录
                confidence_overlay.save(output_path)
                print(f"Processed and saved heatmap for {img_name} at {output_path}")

                return confidence_overlay

            # ------------------------------------------------#
            #   根据mix_type返回最终结果
            # ------------------------------------------------#
            if self.mix_type == 0:
                seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pred_classes, [-1])],
                                     [orininal_h, orininal_w, -1])
                image = Image.fromarray(np.uint8(seg_img))
                image = Image.blend(old_img, image, 0.7)

            elif self.mix_type == 1:
                seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pred_classes, [-1])],
                                     [orininal_h, orininal_w, -1])
                image = Image.fromarray(np.uint8(seg_img))

            elif self.mix_type == 2:
                seg_img = (np.expand_dims(pred_classes != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
                image = Image.fromarray(np.uint8(seg_img))

            return image

    def get_FPS(self, image, test_interval):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------#
                #   图片传入网络进行预测
                # ---------------------------------------------------#
                pr = self.net(images)[0]
                # ---------------------------------------------------#
                #   取出每一个像素点的种类
                # ---------------------------------------------------#
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                # --------------------------------------#
                #   将灰条部分截取掉
                # --------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                     int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

#文件预测路径
dir_origin_path = "./uploads/"

#文件保存路径
path_save = "./img/"
#预测图片保存路径
path_predict_img = "./img_save/"

# 批量图像预测保存路径
batch_image_save_path = "./piliang_img_save/"

#预测视频保存路径
path_predict_video = "./video_save/"
