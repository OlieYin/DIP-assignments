import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def to_tensor_ar(ar, positions):
    tensor = np.zeros((ar.shape[0],positions.shape[0],positions.shape[1],1,ar.shape[1]))

    for i, ai in enumerate(ar):
        tensor[i,:,:] = ai

    return tensor

def to_tensor_m(matrix, positions):
    tensor = np.zeros((positions.shape[0],positions.shape[1],matrix.shape[0],matrix.shape[1]))
    tensor[:,:] = matrix

    return tensor
    


def coefficient_cal(ps, positions, alpha=1.5):
    ps_arr = np.zeros((len(ps), positions.shape[0], positions.shape[1],2))
    omega_arr = np.zeros((len(ps),positions.shape[0],positions.shape[1]))

    for i, pi in enumerate(ps):
        ps_arr[i] = pi

    for i, pi in enumerate(ps_arr):
        omega_arr[i] = 1 / (np.linalg.norm(pi - positions,axis=2) ** (2 * alpha)+1e-15)

    return omega_arr
    


def p_star_cal(ps, positions, omega_arr):
    p_star_arr = np.zeros((positions.shape[0], positions.shape[1],2))
    p_star_arr = np.sum(omega_arr[:,:,:,np.newaxis] * to_tensor_ar(ps, positions)[:,:,:,0,:], axis=0) / np.sum(omega_arr,axis=0)[:,:,np.newaxis]

    return p_star_arr



def q_star_cal(qs, positions, omega_arr):
    q_star_arr = np.zeros((positions.shape[0], positions.shape[1],2))
    q_star_arr = np.sum(omega_arr[:,:,:, np.newaxis] * to_tensor_ar(qs, positions)[:,:,:,0,:], axis=0) / np.sum(omega_arr,axis=0)[:,:,np.newaxis]

    return q_star_arr
    

def p_hat_cal(ps, p_star,positions):
    p_hat_arr = np.zeros((ps.shape[0],positions.shape[0],positions.shape[1],2))

    for i, pi in enumerate(ps):
        p_hat_arr[i] = to_tensor_ar(pi[np.newaxis,:],positions)[0,:,:,0,:] - p_star

    return p_hat_arr



def q_hat_cal(qs, q_star,positions):
    q_hat_arr = np.zeros((qs.shape[0],positions.shape[0],positions.shape[1],2))

    for i, qi in enumerate(qs):
        q_hat_arr[i] = to_tensor_ar(qi[np.newaxis,:],positions)[0,:,:,0,:] - q_star

    return q_hat_arr


def inv_matrix_cal(p_hat_arr, omega_arr):
    p_new_hat_arr = p_hat_arr[:,:,:,np.newaxis,:]
    matrix_arr = np.zeros((omega_arr.shape[0], omega_arr.shape[1], omega_arr.shape[2],2,2))
    inv_matrix_arr = np.zeros((omega_arr.shape[0], omega_arr.shape[1], omega_arr.shape[2],2,2))

    for i, pi in enumerate(p_new_hat_arr):
        matrix_arr[i] =omega_arr[i][:,:,np.newaxis,np.newaxis] * np.transpose(pi,(0,1,3,2)) * pi
    
    inv_matrix_arr = np.linalg.inv(np.sum(matrix_arr,axis=0))

    return inv_matrix_arr
    


def A_matrix_cal(positions, p_star, inv_matrix, omega_arr, p_hat_arr):
    As = np.zeros((omega_arr.shape[0], omega_arr.shape[1], omega_arr.shape[2], 1, 1))

    for i, oi in enumerate(omega_arr):
        As[i,:,:,:,:] = (positions - p_star)[:,:,np.newaxis,:] @ inv_matrix @ np.transpose((oi[:,:,np.newaxis,np.newaxis] * p_hat_arr[i][:,:,np.newaxis,:]),(0,1,3,2))

    return As[:,:,:,0]

def transformed_coords(ps, qs, positions, alpha=1.5):
    transformed_positions_arr = np.zeros((positions.shape[0], positions.shape[1], 2))
    transformed_coords_x = np.zeros(np.shape(positions))
    transformed_coords_y = np.zeros(np.shape(positions))

    omega_arr = coefficient_cal(ps, positions)
    p_star_arr = p_star_cal(ps, positions, omega_arr)
    q_star_arr = q_star_cal(qs, positions, omega_arr)
    p_hat_arr = p_hat_cal(ps, p_star_arr,positions)
    q_hat_arr = q_hat_cal(qs, q_star_arr,positions)
    inv_matrix = inv_matrix_cal(p_hat_arr, omega_arr)
    A_arr = A_matrix_cal(positions, p_star=p_star_arr, inv_matrix=inv_matrix, omega_arr=omega_arr, p_hat_arr=p_hat_arr)

    transformed_positions_arr = np.sum(A_arr * q_hat_arr,axis=0) + q_star_arr
    transformed_coords_x, transformed_coords_y = transformed_positions_arr[:,:,0], transformed_positions_arr[:,:,1]
    
    return transformed_coords_x, transformed_coords_y


def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    y_coords, x_coords = np.indices((warped_image.shape[0],warped_image.shape[1]))
    positions = np.stack((x_coords, y_coords), axis=-1)
    
    ### FILL: 基于MLS or RBF 实现 image warping
    transformed_coords_x, transformed_coords_y = transformed_coords(source_pts, target_pts, positions, alpha=alpha)
    transformed_coords_x = np.clip(transformed_coords_x, 0, warped_image.shape[1]-1).astype(int)
    transformed_coords_y = np.clip(transformed_coords_y, 0, image.shape[0]-1).astype(int)
    warped_image = image[transformed_coords_y, transformed_coords_x]

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_dst), np.array(points_src))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch(share=True)
