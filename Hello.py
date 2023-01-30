import cv2
from keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from streamlit_vertical_slider import vertical_slider
from utils import set_bg

st.set_page_config(page_title="Main", page_icon="📈")
set_bg('images/mnist2.png')

st.sidebar.image('./images/sidebar_main.jpg')

st.image('./images/title.png')

st.write('### :star: :red[특별 체험]')


calculator = st.checkbox('Calculator')
if calculator:
    st.write('좋아요!')
lottery = st.checkbox('Lottery')
if lottery:
    st.image('./images/따봉1.png')
phone_number = st.checkbox('Phone Number')
if phone_number:
    st.image('./images/따봉2.png')


st.write('### :mag: :blue[기본 체험]')

@st.cache(allow_output_mutation=True)         # 모델을 한 번만 load하기 위해서 Rerun을 안하도록 만들어줌
def load():
    return load_model('./model.h5')
model = load()

CANVAS_SIZE = 340

col1, col2 = st.columns(2)

with col1:
    canvas = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas'
    )

if canvas.image_data is not None:
    img = canvas.image_data.astype(np.uint8)
    img = cv2.resize(img, dsize=(28, 28))
    preview_img = cv2.resize(img, dsize=(CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)

    col2.image(preview_img)

    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = x.reshape((-1, 28, 28, 1))
    y = model.predict(x).squeeze()

    st.write('## :dart: 예측 결과: %d' % np.argmax(y))

st.write('### :green[예쁜 사진으로 보기]')
[col3, col4, col5] = st.columns([2, 2, 1])
with col3:
    if st.button('일치'):
        if np.argmax(y) == 0:
            st.image('./images/zero.jpg')
            with open("./images/zero.jpg", "rb") as file:
                btn = st.download_button(
                        label="이미지 소장하기",
                        data=file,
                        file_name="zero.jpg",
                        mime="image/jpg"
                    )
        elif np.argmax(y) == 1:
            st.image('./images/one.jpg')
            with open("./images/one.jpg", "rb") as file:
                btn = st.download_button(
                        label="이미지 소장하기",
                        data=file,
                        file_name="one.jpg",
                        mime="image/jpg"
                    )
        elif np.argmax(y) == 2:
            st.image('./images/two.jpg')
            with open("./images/two.jpg", "rb") as file:
                btn = st.download_button(
                        label="이미지 소장하기",
                        data=file,
                        file_name="two.jpg",
                        mime="image/jpg"
                    )
        elif np.argmax(y) == 3:
            st.image('./images/three.jpg')
            with open("./images/three.jpg", "rb") as file:
                btn = st.download_button(
                        label="이미지 소장하기",
                        data=file,
                        file_name="three.jpg",
                        mime="image/jpg"
                    )
        elif np.argmax(y) == 4:
            st.image('./images/four.jpg')
            with open("./images/four.jpg", "rb") as file:
                btn = st.download_button(
                        label="이미지 소장하기",
                        data=file,
                        file_name="four.jpg",
                        mime="image/jpg"
                    )
        elif np.argmax(y) == 5:
            st.image('./images/five.jpg')
            with open("./images/five.jpg", "rb") as file:
                btn = st.download_button(
                        label="이미지 소장하기",
                        data=file,
                        file_name="five.jpg",
                        mime="image/jpg"
                    )
        elif np.argmax(y) == 6:
            st.image('./images/six.jpg')
            with open("./images/six.jpg", "rb") as file:
                btn = st.download_button(
                        label="이미지 소장하기",
                        data=file,
                        file_name="six.jpg",
                        mime="image/jpg"
                    )
        elif np.argmax(y) == 7:
            st.image('./images/seven.jpg')
            with open("./images/seven.jpg", "rb") as file:
                btn = st.download_button(
                        label="이미지 소장하기",
                        data=file,
                        file_name="seven.jpg",
                        mime="image/jpg"
                    )
        elif np.argmax(y) == 8:
            st.image('./images/eight.jpg')
            with open("./images/eight.jpg", "rb") as file:
                btn = st.download_button(
                        label="이미지 소장하기",
                        data=file,
                        file_name="eight.jpg",
                        mime="image/jpg"
                    )
        elif np.argmax(y) == 9:
            st.image('./images/nine.jpg')
            with open("./images/nine.jpg", "rb") as file:
                btn = st.download_button(
                        label="이미지 소장하기",
                        data=file,
                        file_name="nine.jpg",
                        mime="image/jpg"
                    )

with col4:
    if st.button('불일치'):
        st.image('./images/불일치.png')
        st.write('#### 다시 그려주세요')
        
with col5:
    st.write("##### 내 만족도")
    vertical_slider(
        key="slider",
        default_value=10,
        step=1,
        min_value=0,
        max_value=100,
        track_color="gray",
        thumb_color="blue", 
        slider_color="red",
    )
    
st.write("### :bluepurple[그래프로 보기]")
st.write("해당 숫자일 확률을 나타냅니다.")
# st.bar_chart(y)

st.write('### BGM :musical_score:')
st.write('Fantasie Impromptu')
st.audio('./audios/Fantasie Impromptu.mp3')
st.write('He is a Perate')
st.audio('./audios/He is a Perate.mp3')
st.write('Merry Christmas Mr-Laurence')
st.audio('./audios/Merry Christmas Mr-Laurence.mp3')
st.write('River flows in your')
st.audio('./audios/River flows in your.mp3')
st.write('Summer')
st.audio('./audios/Summer.mp3')