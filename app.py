import streamlit as st
import json
from datetime import datetime

# Navigation
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "About Me", "Projects", "Skills", "Work Experience", "Contact"])

    if selection == "Home":
        home()
    elif selection == "About Me":
        about_me()
    elif selection == "Projects":
        projects()
    elif selection == "Skills":
        skills()
    elif selection == "Work Experience":
        work_experience()
    elif selection == "Contact":
        contact()

# Home Page
def home():
    st.title("Welcome to My Portfolio")
    st.write("""
    I am a third-year student currently pursuing a degree in Artificial Intelligence and Data Science. 
    I have expertise in Data Science, Machine Learning, Natural Language Processing (NLP),Deep Learniinig and Computer Vision, 
    and I am currently learning MLOps.
    """)
    st.write("Feel free to explore my projects and learn more about my skills and experiences.")

# About Me Page
def about_me():
    st.title("About Me")
    st.write("""
    Hello! I'm Aditya Kumar, a third-year student currently pursuing a degree in Artificial Intelligence and Data Science. 
    I have a passion for technology and a keen interest in Data Science, Machine Learning, Natural Language Processing (NLP), 
    and Computer Vision. Currently, I'm expanding my knowledge in MLOps.
    """)
    
    # Display resume
    with open("resume.pdf", "rb") as file:
        st.download_button(label="Download Resume", data=file, file_name="resume.pdf", mime="application/pdf")
    
    # Display social media links
    st.write("Connect with me:")
    st.markdown("""
    - [LinkedIn](https://www.linkedin.com/in/aditya-kumar-8a537528a/)
    - [GitHub](https://github.com/Aditya-professional-life)
    - [Twitter](https://twitter.com/KumarAdity99116/with_replies)
    """)

# Projects Page
def projects():
    st.title("Projects")
    
    # Project 1: LangChain PDF Interact
    st.header("LangChain PDF Interact")
    st.write("""
    - **Technologies:** Streamlit, PyPDF2, langchain, Hugging Face Transformers, dotenv, FAISS
    - **Description:** Developed a web application using Streamlit and Python for PDF processing.
      Integrated langchain NLP libraries and Hugging Face Transformers for natural language understanding.
      Utilized dotenv for environment variables and FAISS for efficient similarity search.
      Facilitated interactive conversational queries on PDF documents.
    """)
    st.markdown("[GitHub Repository](https://github.com/Aditya-professional-life/Chat-with-multiple-pdfs)")

    # Project 2: Human Activity Recognition
    st.header("Human Activity Recognition")
    st.write("""
    - **Technologies:** Scikit-learn, XGBoost, TensorFlow, Collections, Regular Expression
    - **Description:** Leading the development of advanced machine learning models to accurately recognize human activities based on sensor data.
      Proficient in Python for comprehensive data preprocessing, exploratory analysis, and model tuning.
      Specialized expertise in implementing logistic regression, decision trees, and random forests.
    """)
    st.markdown("[GitHub Repository](https://github.com/Aditya-professional-life/Human-Activity-Recognition-system)")

    # Project 3: Real-time Object Detection & Tracking
    st.header("Real-time Object Detection & Tracking")
    st.write("""
    - **Technologies:** OpenCV, Tracker
    - **Description:** Engineered real-time object tracking with Python, OpenCV, and Euclidean distance tracking.
      Applied advanced computer vision techniques like background subtraction and contour detection.
      Demonstrated Python proficiency and algorithmic optimization to enhance object tracking performance.
    """)
    st.markdown("[GitHub Repository](https://github.com/Aditya-professional-life/Real-Time-Object-Tracking-System)")

# Skills Page
def skills():
    st.title("Skills")
    
    # Programming Skills
    st.header("Programming")
    st.write("""
    - Python
    - SQL
    - Numpy
    - Matplotlib
    - Seaborn
    - Streamlit
    - Selenium
    - Statistics
    """)

    # Machine Learning Skills
    st.header("Machine Learning")
    st.write("""
    - Preprocessing
    - NA handling
    - Outlier treatment
    - Data normalization
    - One hot encoding
    - Feature engineering
    - Cross validation
    - Model Building
    - Supervised Learning
    - Unsupervised Learning
    """)

    # Image Processing Skills
    st.header("Image Processing")
    st.write("""
    - OpenCV
    - Computer Vision
    - Digital Image Processing
    - Webcam Processing
    - Face Detection
    - Real-time Processing
    """)
import streamlit as st

# Work Experience Page
def work_experience():
    st.title("Work Experience")
    
    # Experience 1: Fashion Cloth Search App
    st.header("Fashion Cloth Search App")
    st.write("""
    - **Present**
    - **Description:** Developed a multimodal search system using Haystack to retrieve fashion item images based on text queries.
    - **Technologies:** In-Memory Document Store, CLIP model, Streamlit
    - **Details:**
        - Implemented an In-Memory Document Store to store image embeddings and utilized the CLIP model for text-to-image retrieval.
        - Created an intuitive user interface with Streamlit, allowing users to enter queries and view the top matching images with similarity scores.
        - Enabled efficient and accurate search functionality, leveraging the latest advancements in multimodal embeddings and interactive web applications.
    """)

    # Experience 2: Ineuron.ai Machine Learning Intern
    st.header("Ineuron.ai Machine Learning Intern")
    st.write("""
    - **Sep 2023 – Oct 2023**
    - **Description:** Conducted comprehensive Exploratory Data Analysis (EDA) on the Zomato Dataset.
    - **Technologies:** Various machine learning algorithms
    - **Details:**
        - Developed and implemented machine learning models for accurate rating prediction.
        - Utilized diverse algorithms to tailor a robust predictive model specific to the Zomato Dataset.
        - Enhanced accuracy in restaurant rating forecasts, contributing to strategic business initiatives.
    """)

    # Experience 3: CodeClause Data Science Intern
    st.header("CodeClause Data Science Intern")
    st.write("""
    - **March 2024 – Present**
    - **Description:** Implemented K-Means clustering for Customer Segmentation.
    - **Technologies:** Statistical methods, K-Means clustering
    - **Details:**
        - Utilized statistical methods to categorize customer data effectively.
        - Employed statistical analysis to drive customer insights.
        - Applied statistical techniques, specifically K-Means clustering, to derive meaningful customer segments.
        - Facilitated targeted business strategies.
    """)
def save_contact(email, message):
    contact_data = {
        "email": email,
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        with open("contacts.json", "r+") as file:
            data = json.load(file)
            data.append(contact_data)
            file.seek(0)
            json.dump(data, file, indent=4)
    except FileNotFoundError:
        with open("contacts.json", "w") as file:
            json.dump([contact_data], file, indent=4)

# Contact Page
def contact():
    st.title("Contact")
    st.write("Get in touch with me by filling out the form below.")
    
    with st.form(key='contact_form'):
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            save_contact(email, message)
            st.success("Your message has been sent!")

if __name__ == "__main__":
    main()
