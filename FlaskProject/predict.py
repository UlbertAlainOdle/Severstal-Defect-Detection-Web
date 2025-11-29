# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from segformer import SegFormer_Segmentation

if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到generate函数里修改self.colors即可
    # -------------------------------------------------------------------------#
    segformer = SegFormer_Segmentation()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "confidence"
    # -------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    count = True
    name_classes = ["background", "patches", "inclusion",  "scratches", "other", "none0", "none1", "none2", "none3",
                    "none4", "none5", "none6", "none7", "none8", "none9", "none10", "none11", "none12", "none13",
                    "none14", "none15"]
    # name_classes    = ["background","cat","dog"]
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #
    #   test_interval和fps_image_path仅在mode='fps'有效
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "test_images"
    dir_save_path = "img_save"
    # -------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        import os
        from tqdm import tqdm
        from PIL import Image, ImageEnhance, ImageFilter
        import numpy as np

        # 获取文件夹中所有图片文件
        img_names = os.listdir(dir_origin_path)


        # 边缘增强函数
        def enhance_edges(image, radius=3, percent=150, threshold=2):
            """增强图像边缘"""
            return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


        # 遍历文件夹中的每个图片，进行预测
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)  # 获取图片路径
                image = Image.open(image_path)  # 打开图片

                # 增强边缘并进行预测
                enhanced_image = enhance_edges(image, radius=3, percent=150, threshold=2)  # 边缘增强
                r_image_enhanced = segformer.detect_image(enhanced_image, count=count, name_classes=name_classes,
                                                          confidence_threshold=0.5, show_confidence=False,
                                                          mode="predict")

                # 将结果转换为numpy数组并保存
                r_image_enhanced_array = np.array(r_image_enhanced)  # 将预测结果转换为numpy数组

                # 保存边缘增强后的结果
                output_path = os.path.join(dir_save_path, img_name)
                final_result_image = Image.fromarray(r_image_enhanced_array.astype(np.uint8))  # 转换为PIL图像
                final_result_image.save(output_path)  # 保存预测结果
                print(f"Processed {img_name}")  # 打印当前处理的文件名

        print("Batch prediction finished.")  # 所有图片处理完成

    elif mode == "heatmap":
        import os
        from tqdm import tqdm

        # 获取文件夹中所有图片文件
        img_names = os.listdir(dir_origin_path)

        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)

                # 进行热力图预测
                r_image = segformer.detect_image(image, count=False, name_classes=name_classes,
                                                 confidence_threshold=0.6, show_confidence=False, mode="heatmap")

                # 确保目标文件夹存在
                output_dir = "D:/aminuos/segformer-pytorch-master/test_results"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)  # 创建文件夹

                # 构建保存路径，去掉heatmap_前缀
                output_path = os.path.join(output_dir, f"{img_name}")

                # 确保添加扩展名
                if not output_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    output_path += '.jpg'  # 默认保存为 .jpg

                r_image.save(output_path)  # 保存热力图结果
                print(f"Processed and saved heatmap for {img_name} at {output_path}")

        print("Heatmap generation finished.")  # 所有图片处理完成

    elif mode == "confidence":
        import os
        from tqdm import tqdm
        from PIL import Image
        import numpy as np

        # 获取文件夹中所有图片文件
        img_names = os.listdir(dir_origin_path)

        # 遍历文件夹中的每个图片，进行预测并计算置信度
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)  # 获取图片路径
                image = Image.open(image_path)  # 打开图片

                # 进行预测（包括置信度计算）
                r_image = segformer.detect_image(image, count=False, name_classes=name_classes,
                                                 confidence_threshold=0.6, show_confidence=True, mode="confidence")

                # 将结果转换为numpy数组并保存
                r_image_array = np.array(r_image)  # 将预测结果转换为numpy数组

                # 保存预测结果
                output_path = os.path.join(dir_save_path, img_name)
                final_result_image = Image.fromarray(r_image_array.astype(np.uint8))  # 转换为PIL图像
                final_result_image.save(output_path)  # 保存预测结果
                print(f"Processed {img_name}")  # 打印当前处理的文件名

        print("confidence mode prediction finished.")  # 所有图片处理完成


    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(segformer.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = segformer.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = segformer.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    elif mode == "export_onnx":
        segformer.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
