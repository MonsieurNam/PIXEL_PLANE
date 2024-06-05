# import time
import streamlit as st
from PIL import Image
import albumentations as A
import cv2
import numpy as np
from edit_image.web_app_ver2 import EditImage

# from change_background_model import model
# from remove_object import *
from img2vid import *

#---------------Xử lý nền------------------#

def image_resize(img, h, w):
  # img = cv2.imread(link)
  img = np.array(img)
  transform = A.Resize(h, w, interpolation=cv2.INTER_NEAREST)
  aug = transform(image=img)
  img = aug['image']
  img_org = Image.fromarray(img)
  return img_org

logo_fpt = Image.open('images/logo_fpt.png')
logo_fpt = image_resize(logo_fpt, int(logo_fpt.height*0.27), int(logo_fpt.width*0.27))

logo_hackathon = Image.open('images/logo_hackathon.png')
logo_hackathon = image_resize(logo_hackathon, int(logo_hackathon.height*1.4), int(logo_hackathon.width*1.4))

logo_donvi_tc = Image.open('images/logo_dvtc.png')
logo_donvi_tc = image_resize(logo_donvi_tc, int(logo_donvi_tc.height*0.9), int(logo_donvi_tc.width*0.9))


#-------------------Frames---------------------#

def side_bar():
# Tạo các nút nằm ngang cho taskbar
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("MAIN MENU"):
            st.session_state.page = "main"
    with col2:
        if st.button("CHANGE BACKGROUND"):
            st.session_state.page = "change_bg"
    with col3:
        if st.button("IMAGE TO VIDEO"):
            st.session_state.page = "img2vid"
    with col4:
        if st.button("EDIT IMAGE"):
            st.session_state.page = "edit_image"
  

def change_background():
    st.text(" ")
    st.markdown("<h1 style='text-align: center; color: white;'>Change Image's Background</h1>", unsafe_allow_html=True)

    st.subheader('Image')
    upload_image = st.file_uploader('Upload the image you want to generate here: ')

    if upload_image is not None:
        # Lưu trữ file ảnh vào session_state
        st.session_state.upload_image = upload_image
    _, center, __ = st.columns(3)
    if 'upload_image' in st.session_state:
        with center:
            st.image(st.session_state.upload_image, caption='Original Image')


    st.subheader('Prompt')
    st.session_state.prompt = st.text_input('Enter your prompt here: ')
    st.session_state.num_img = st.number_input('Number of images you want to create: ', 1, 4)

    if st.button('Generate'):
        # columns = st.columns(st.session_state.num_img)
        # with st.spinner('Generating...'):
            st.session_state.upload_image = Image.open(st.session_state.upload_image)
        #     output_images = model(image=st.session_state.upload_image, prompt=st.session_state.prompt, num=st.session_state.num_img)
        #
        #     for i, img in enumerate(output_images):
        #         with columns[i]:
        #             st.image(img, use_column_width=True)
        # st.success('Done!')
            _, center, __ = st.columns(3)
            with center:
                st.image(st.session_state.upload_image)
                st.write('aaaaaaaaaaaaaaaaaaaaaaaaa')


def img2vid():
    st.subheader('Image')
    upload_image = st.file_uploader('Upload the image you want to generate here: ')

    if upload_image is not None:
        # Lưu trữ file ảnh vào session_state
        st.session_state.upload_image = upload_image

    if 'uploaded_image' in st.session_state:
        st.image(st.session_state.upload_image, caption='Original Image.')

    if st.button('Generate'):
        video = image2video(st.session_state.upload_image)
        st.video(video)



#
def edit_image():
    EditImage()




#-----------main page code-------------#
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .stButton button {
        font-size: 50px;
        padding: 20px 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .center {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

lg1, lg2, lg3 = st.columns(3)
with lg1:
    st.image(logo_fpt)
with lg2:
    st.image(logo_hackathon)
with lg3:
    st.image(logo_donvi_tc)

# Khởi tạo state cho trang hiện tại nếu chưa có
if 'page' not in st.session_state:
    st.session_state.page = "main"

# Hàm để hiển thị nội dung của từng trang
def show_page(page):
    if page == "main":
        side_bar()
    elif page == "change_bg":
        side_bar()
        change_background()
    elif page == "img2vid":
        side_bar()
        st.header("Image to Video")
        # img2vid()
    elif page == "edit_image":
        side_bar()
        edit_image()


# Hiển thị trang hiện tại
show_page(st.session_state.page)

