import cv2
import os


def get_image_resolutions(folder_path):
    max_resolution = (0, 0)
    min_resolution = (float('inf'), float('inf'))

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is not None:
                height, width = img.shape[:2]
                resolution = (width, height)

                if resolution[0] * resolution[1] > max_resolution[0] * max_resolution[1]:
                    max_resolution = resolution

                if resolution[0] * resolution[1] < min_resolution[0] * min_resolution[1]:
                    min_resolution = resolution

    return max_resolution, min_resolution


# 替换为您的图片文件夹路径
folder_path = r'E:\ImagefusionDatasets\StackMFF\Mobile Depth\AiF'

max_res, min_res = get_image_resolutions(folder_path)

print(f"Maximum resolution: {max_res[0]}x{max_res[1]}")
print(f"Minimum resolution: {min_res[0]}x{min_res[1]}")