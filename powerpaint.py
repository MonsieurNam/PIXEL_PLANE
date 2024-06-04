import streamlit as st
import torch
from PIL import Image, ImageOps
import numpy as np
from diffusers.utils import load_image
from io import BytesIO

# Function to add tasks to prompt
def add_task_to_prompt(prompt, negative_prompt, task):
    if task == "object-removal":
        promptA = prompt + " P_ctxt"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt + " P_obj"
        negative_promptB = negative_prompt + " P_obj"
    elif task == "shape-guided":
        promptA = prompt + " P_shape"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
    elif task == "image-outpainting":
        promptA = prompt + " P_ctxt"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt + " P_obj"
        negative_promptB = negative_prompt + " P_obj"
    else:
        promptA = prompt + " P_obj"
        promptB = prompt + " P_obj"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
    return promptA, promptB, negative_promptA, negative_promptB

# Main processing function
@torch.inference_mode()
def predict(
    pipe,
    input_image,
    prompt,
    fitting_degree,
    ddim_steps,
    scale,
    negative_prompt,
    task,
):
    # Adjusting image size based on aspect ratio
    width, height = input_image["image"].convert("RGB").size
    if width < height:
        input_image["image"] = (
            input_image["image"].convert("RGB").resize((640, int(height / width * 640)))
        )
    else:
        input_image["image"] = (
            input_image["image"].convert("RGB").resize((int(width / height * 640), 640))
        )

    promptA, promptB, negative_promptA, negative_promptB = add_task_to_prompt(
        prompt, negative_prompt, task
    )
    img = np.array(input_image["image"].convert("RGB"))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image["image"] = input_image["image"].resize((H, W))
    input_image["mask"] = input_image["mask"].resize((H, W))

    result = pipe(
        promptA=promptA,
        promptB=promptB,
        tradoff=fitting_degree,
        tradoff_nag=fitting_degree,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=input_image["image"].convert("RGB"),
        mask_image=input_image["mask"].convert("RGB"),
        guidance_scale=scale,
        num_inference_steps=ddim_steps,
    ).images[0]
    return result

# Load the model
def gen_image(pipe, uploaded_image,uploaded_mask,prompt,guidance_scale,negative_prompt,task):
    pipe = pipe
    image = uploaded_image
    mask = uploaded_mask
    input_image = {"image": image, "mask": mask}
    result_image = predict(
        pipe,
        input_image,
        prompt,
        1,  # fitting_degree
        30,  # ddim_steps
        guidance_scale,
        negative_prompt,
        task,
    )
    return result_image