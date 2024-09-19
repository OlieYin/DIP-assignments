import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

def scale_matrix(scale=1,trans_x=0,trans_y=0):
    return np.array([[1/scale,0,trans_x*(1-1/scale)],
                    [0,1/scale,trans_y*(1-1/scale)]])

def rotation(theta=0):
    deg = theta*2*np.pi/360
    return np.array([[np.cos(deg),-np.sin(deg),0],
                    [np.sin(deg),np.cos(deg),0]])

def rotation_matrix(theta,trans_x,trans_y):
    return np.array(to_3x3(translation_matrix(trans_y,trans_x)) @ to_3x3(rotation(theta)) @ to_3x3(translation_matrix(-trans_y,-trans_x)))

def translation_matrix(translation_x=0,translation_y=0):
    return np.array([[1,0,translation_y],
                    [0,1,translation_x]])

def flip_horizontal(is_flip, width):
    if is_flip == 0:
        return np.array([[1,0,0],
                         [0,1,0]])
    elif is_flip ==1:
        return np.array([[1,0,0],
                         [0,-1,width]])


def comp_matrix(scale, rotation, translation_x, translation_y,trans_x,trans_y, is_flip, width):
    return   np.array(to_3x3(translation_matrix(translation_x, translation_y)) @ rotation_matrix(rotation,trans_x,trans_y) @ to_3x3(scale_matrix(scale,trans_x,trans_y))) @ to_3x3(flip_horizontal(is_flip, width))


# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, is_flip):
    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size_x = image.shape[0]// 2
    pad_size_y = image.shape[1]//2
    image_new = np.zeros((pad_size_x*2+image.shape[0], pad_size_y*2+image.shape[1], 3), dtype=np.uint8) + np.array((100,100,100), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size_x:pad_size_x+image.shape[0], pad_size_y:pad_size_y+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)
    trans_image_x, trans_image_y = transformed_image.shape[0], transformed_image.shape[1]

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    x_coords, y_coords = np.indices((trans_image_x,trans_image_y))
    ones = np.ones((trans_image_x,trans_image_y))
    homo_coords = np.stack([x_coords,y_coords,ones],axis=-1)
    transformed_coords = homo_coords @ comp_matrix(scale, rotation, translation_x, translation_y,trans_x=2*pad_size_x,trans_y=2*pad_size_y, is_flip=is_flip, width=trans_image_y).T
    transformed_coords_x, transformed_coords_y = np.clip(transformed_coords[...,0]/transformed_coords[...,2],0,trans_image_x-1).astype(int), np.clip(transformed_coords[...,1]/transformed_coords[...,2],0,trans_image_y-1).astype(int)


    return transformed_image[transformed_coords_x,transformed_coords_y]


# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch(share=True)
