#install dependence
!pip install -r requirement.txt
%cd /content/segment_anything
!pip install -q .
! wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

#run
!wget -q -O - ipv4.icanhazip.com
!streamlit run web_app_ver2.py --server.enableXsrfProtection false &>/dev/null & npx localtunnel --port 8501


#reset gpu
import torch
torch.cuda.empty_cache()