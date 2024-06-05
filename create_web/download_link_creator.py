import io
import base64
from PIL import Image
import streamlit as st

def get_image_download_link(image, filename, text):
    """Generates a link to download a particular image file."""
    img =Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    st.write(href)
    # return href
