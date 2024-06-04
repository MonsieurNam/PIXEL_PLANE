from base64 import b64encode
import os
import io
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

from GroundingDINO.groundingdino.util.inference import annotate, predict
from PIL import Image, ImageOps
import requests
import streamlit as st
import torch

# Grounding DINO
from detect_DINO import detect, groundingdino_model, load_image
#sam
from sam import segment, draw_mask, sam_predictor
#mask create
from mask_create import create_mask
#model diffusers
from powerpaint import gen_image
from pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from power_paint_tokenizer import PowerPaintTokenizer

@st.cache_resource
def load_model_gen():
  pipe = Pipeline.from_pretrained(
      "Sanster/PowerPaint-V1-stable-diffusion-inpainting",
      torch_dtype=torch.float16,
      safety_checker=None,
      variant="fp16",
  )
  pipe.tokenizer = PowerPaintTokenizer(pipe.tokenizer)
# Automatically select the device
  if torch.cuda.is_available():
      return  pipe.to("cuda")
  else:
      return  pipe.to("cpu")
  
pipe = load_model_gen()  
st.title('Image detect+segment+inpaint')
image_upload = st.file_uploader("Upload a photo")
mask_creation_method = st.radio("Choose the method to create a mask:", ('Use Prompt', 'Draw Mask'))

if image_upload is None:
    st.stop()

if 'image_mask_pil' not in st.session_state:
    st.session_state['image_mask_pil'] = None

if image_upload is not None:
    st.session_state.image_source, image = load_image(image_upload)
    st.subheader('Image orginal')
    st.image(image_upload)

    if mask_creation_method == 'Use Prompt':
        prompt_chosse_object = st.text_input(label="Describe the object you want to change:", key="prompt_object")
        if prompt_chosse_object:
            annotated_frame, detected_boxes = detect(st.session_state.image_source, image,text_prompt=prompt_chosse_object, model=groundingdino_model)

            # st.subheader('Result of detect')
            # st.image(annotated_frame, caption = prompt_chosse_object)

            #segmnet base on detect object
            if detected_boxes.nelement() !=0:
                segmented_frame_masks = segment(st.session_state.image_source, sam_predictor, boxes=detected_boxes)
                annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated_frame)

                st.subheader('Result of segment')
                st.image(annotated_frame_with_mask)

                #creat mask         
                st.session_state.image_source_pil, st.session_state.image_mask_pil, inverted_image_mask_pil = create_mask(st.session_state.image_source, segmented_frame_masks)

                st.subheader('Result of mask')
                st.image(st.session_state.image_mask_pil)


    elif mask_creation_method == 'Draw Mask':
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 5)
        h, w = st.session_state.image_source.shape[:2]
        scale_factor = 800 / max(w, h) if max(w, h) > 800 else 1
        w_, h_ = int(w * scale_factor), int(h * scale_factor)

        # Create a canvas component.
        st.subheader('Draw mask')
        canvas_result = st_canvas(
            fill_color='rgba(255, 255, 255, 0)',
            stroke_width=stroke_width,
            stroke_color='white',
            background_image=Image.fromarray(st.session_state.image_source).resize((w_, h_)),
            update_streamlit=True,
            height=h_,
            width=w_,
            drawing_mode='freedraw',
            key="canvas",
        )

        if canvas_result.image_data is not None:
            mask = cv2.cvtColor(np.array(canvas_result.image_data), cv2.COLOR_RGBA2GRAY)
            mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Process the mask into the required format
            st.session_state.image_source_pil, st.session_state.image_mask_pil, inverted_image_mask_pil = create_mask(st.session_state.image_source, mask)
            
            st.subheader('Result of mask')
            st.image(st.session_state.image_mask_pil, caption='Generated Mask')

        # Inpainting form
    if st.session_state.image_mask_pil is not None:
            with st.form("Prompt"):
                st.session_state.prompt = st.text_input(label="What would you like to see replaced?")
                task = "inpainting"
                st.session_state.guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5)
                st.session_state.negative_prompt = "out of frame, lowres, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, watermark, signature"
                submitted = st.form_submit_button("Generate")
                if submitted:
                    result_image = gen_image(pipe, st.session_state.image_source_pil,st.session_state.image_mask_pil, st.session_state.prompt,st.session_state.guidance_scale,st.session_state.negative_prompt,task)
                    st.image(result_image, caption="Processed Image")



  
                            

