import streamlit as st
from PIL import Image
import numpy as np
import requests
from datetime import datetime
import io
import base64
import bson.binary
import warnings

warnings.filterwarnings("ignore")

# Optional: MongoDB setup (commented)
# from pymongo import MongoClient
# from dotenv import load_dotenv
# import os
# load_dotenv()
# MONGO_URI = os.getenv("MONGO_URI")
# client = MongoClient(MONGO_URI)
# db = client['iihmr']
# ic = db['patientinfo']


def get_current_date():
    try:
        response = requests.get("http://worldtimeapi.org/api/ip", timeout=5)
        if response.status_code == 200:
            try:
                data = response.json()
                datetime_str = data["datetime"]
                date, time = datetime_str.split("T")
                time = time.split(".")[0]
                return date, time
            except ValueError:
                st.warning("⚠️ Failed to parse time from API. Using system time.")
        else:
            st.warning(f"⚠️ Time API returned status code {response.status_code}. Using system time.")
    except Exception as e:
        st.warning(f"⚠️ Time API request failed. Using system time. Error: {e}")

    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")


def process_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    b, g, r = cv2.split(image_np)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_b = clahe.apply(b)
    clahe_g = clahe.apply(g)
    clahe_r = clahe.apply(r)

    image_clahe = cv2.merge((clahe_b, clahe_g, clahe_r))
    image_clahe = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2RGB)
    return image, image_clahe


def work(img):
    image = Image.fromarray(img)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)

    img_data = base64.b64encode(buffer.read()).decode("utf-8")

    try:
        response = requests.post("http://127.0.0.1:5001/predict", json={"file": img_data}, timeout=10)
        if response.status_code == 200:
            try:
                res = response.json()
                labels = ['Oral Cancer', 'No Abnormality detected', 'Oral premalignant lesion']
                for i in range(3):
                    st.subheader(f"Probability of class '{labels[i]}': {res[i]:.1f}%")
                return res
            except Exception as e:
                st.error(f"⚠️ Failed to parse response from model. Error: {e}")
        else:
            st.error(f"⚠️ Model API returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Could not connect to model API. Is the server running? Error: {e}")

    return [0, 0, 0]


def main():
    image_url = "https://i.imgur.com/7ppJJP4.jpeg"

    st.markdown(f"""
        <style>
            .center {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100;
            }}
            .image-container {{
                text-align: center;
            }}
        </style>
        <div class="center">
            <div class="image-container">
                <img src="{image_url}" style="width: 200px;">
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center;'>
        <h2 style='font-size: 2em;'>Primary Referral and AI-based Screening for Assessing Oral Health</h2>
    </div>
    """, unsafe_allow_html=True)

    name = st.text_input(label="Enter Name:\n:red[*]")
    gender = st.selectbox(label="Select gender:\n:red[*]", options=("None", "Male", "Female", "Other"))
    age = st.number_input(label="Enter age:\n:red[*]", min_value=0, max_value=120, step=1)
    d, t = get_current_date()
    abha = st.text_input("Enter ABHA ID:")

    sub = False
    image = None
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

    if st.button("Submit"):
        if not name:
            st.error("Patient's name is required.")
        elif gender == "None":
            st.error("Patient's gender can't be None")
        elif age is None or age <= 0:
            st.error("Patient's age is required.")
        else:
            sub = True

    if image is not None and sub:
        original_image, processed_image = process_image(image)

        st.image(original_image, caption='Enhanced Image', width=224)

        if not abha:
            abha = "None"

        result = work(processed_image)

        resized_image = image.resize((224, 224))
        img_byte_arr = io.BytesIO()
        resized_image.save(img_byte_arr, format='PNG')
        image_binary = bson.Binary(img_byte_arr.getvalue())

        document = {
            'name': name,
            'gender': gender,
            'age': age,
            'abha': abha,
            'date': d,
            'time': t,
            'cancer': result[0],
            'normal': result[1],
            'OPMD': result[2],
            'image_data': image_binary
        }

        # ic.insert_one(document)
        # st.success("✅ Patient record uploaded successfully.")


if __name__ == "__main__":
    main()

