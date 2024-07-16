import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

@st.cache_data
def load_data():
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target
    X = X.astype('float32') / 255.
    y = y.astype('int')
    return X, y

def main():
    st.set_page_config(page_title="Data Augmentation Techniques Demo", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    .stButton>button {
        background-color: #ff6e40;
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff9e80;
    }
    .quiz-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🖼️ Data Augmentation Techniques in Machine Learning")
    
    tabs = st.tabs(["📚 Learn", "🧪 Experiment", "📊 Visualization", "🧠 Quiz"])
    
    with tabs[0]:
        learn_section()
    
    with tabs[1]:
        experiment_section()
    
    with tabs[2]:
        visualization_section()
    
    with tabs[3]:
        quiz_section()

def learn_section():
    st.header("Understanding Data Augmentation")
    
    st.write("""
    Data augmentation is a technique used to increase the amount of training data by creating modified versions of existing data. 
    It's particularly useful in image processing and computer vision tasks, but can be applied to other types of data as well.

    Key benefits of data augmentation include:
    """)

    benefits = [
        "Increasing the size of the training dataset",
        "Reducing overfitting",
        "Improving model generalization",
        "Helping models learn invariance to certain transformations"
    ]

    for benefit in benefits:
        st.write(f"- {benefit}")

    st.subheader("Common Data Augmentation Techniques for Images:")
    techniques = {
        "Rotation": "Rotating the image by a certain angle",
        "Flipping": "Flipping the image horizontally or vertically",
        "Scaling": "Zooming in or out of the image",
        "Translation": "Moving the image along the X or Y axis",
        "Adding Noise": "Adding random noise to the image",
        "Changing Brightness": "Adjusting the brightness of the image",
        "Cropping": "Randomly cropping parts of the image"
    }

    for technique, description in techniques.items():
        st.write(f"**{technique}**: {description}")

    st.write("""
    Data augmentation is crucial in machine learning because:
    - It helps models learn to be invariant to certain transformations
    - It can significantly improve performance when training data is limited
    - It helps in creating more robust models that generalize better to unseen data
    """)

def experiment_section():
    st.header("🧪 Experiment with Data Augmentation")

    X, y = load_data()

    st.subheader("Original MNIST Data")
    st.write("Shape of the data:", X.shape)
    st.write("Number of classes:", len(np.unique(y)))

    # Display a random original image
    index = np.random.randint(0, X.shape[0])
    st.image(X[index].reshape(28, 28), caption="Original Image", width=200)

    # Data Augmentation Options
    st.subheader("Select Data Augmentation Techniques")
    
    rotation = st.checkbox("Rotation")
    if rotation:
        rotation_range = st.slider("Rotation range", 0, 180, 20)

    flip = st.checkbox("Horizontal Flip")
    
    zoom = st.checkbox("Zoom")
    if zoom:
        zoom_range = st.slider("Zoom range", 0.0, 1.0, 0.2)

    brightness = st.checkbox("Brightness Adjustment")
    if brightness:
        brightness_range = st.slider("Brightness range", 0.0, 1.0, 0.2)

    # Apply selected augmentation techniques
    datagen = ImageDataGenerator(
        rotation_range=rotation_range if rotation else 0,
        horizontal_flip=flip,
        zoom_range=zoom_range if zoom else 0,
        brightness_range=(1-brightness_range, 1+brightness_range) if brightness else None
    )

    # Reshape the image for augmentation
    img = X[index].reshape((1, 28, 28, 1))

    # Generate augmented images
    aug_iter = datagen.flow(img, batch_size=1)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        image = next(aug_iter)[0].astype('float32')
        ax[i].imshow(image.reshape(28, 28), cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(f"Augmented Image {i+1}")
    
    st.pyplot(fig)

    st.write("""
    Experiment with different augmentation techniques to see how they transform the original image.
    These transformations can help the model learn to recognize digits regardless of small variations.
    """)

def visualization_section():
    st.header("📊 Visualizing Data Augmentation Effects")

    X, y = load_data()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model without augmentation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy_without_aug = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance without Augmentation")
    st.write(f"Accuracy: {accuracy_without_aug:.4f}")

    # Train with augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2
    )

    # Reshape the data for augmentation
    X_train_reshaped = X_train.reshape(X_train.shape[0], 28, 28, 1)

    # Create an augmented dataset
    aug_data = []
    aug_labels = []
    for X_batch, y_batch in datagen.flow(X_train_reshaped, y_train, batch_size=32):
        aug_data.append(X_batch)
        aug_labels.append(y_batch)
        if len(aug_data) * 32 >= X_train.shape[0]:
            break

    X_train_aug = np.vstack([X_train] + [x.reshape(x.shape[0], -1) for x in aug_data])
    y_train_aug = np.concatenate([y_train] + aug_labels)

    # Train the model with augmented data
    X_train_aug_scaled = scaler.fit_transform(X_train_aug)
    model_aug = LogisticRegression(random_state=42)
    model_aug.fit(X_train_aug_scaled, y_train_aug)
    y_pred_aug = model_aug.predict(X_test_scaled)
    accuracy_with_aug = accuracy_score(y_test, y_pred_aug)

    st.subheader("Model Performance with Augmentation")
    st.write(f"Accuracy: {accuracy_with_aug:.4f}")

    # Visualize the difference
    accuracies = [accuracy_without_aug, accuracy_with_aug]
    labels = ['Without Augmentation', 'With Augmentation']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=labels, y=accuracies, ax=ax)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    st.pyplot(fig)

    st.write("""
    This visualization shows the impact of data augmentation on model performance.
    Data augmentation typically leads to improved accuracy and better generalization.
    """)

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main purpose of data augmentation?",
            "options": [
                "To reduce the size of the dataset",
                "To increase the variety of training data",
                "To remove outliers from the dataset",
                "To normalize the features"
            ],
            "correct": "To increase the variety of training data",
            "explanation": "Data augmentation aims to increase the diversity of the training data by creating modified versions of existing data. This helps in improving model generalization and reducing overfitting."
        },
        {
            "question": "Which of the following is NOT a common image augmentation technique?",
            "options": [
                "Rotation",
                "Flipping",
                "Changing color channels",
                "Adding new objects to the image"
            ],
            "correct": "Adding new objects to the image",
            "explanation": "While rotation, flipping, and changing color channels are common augmentation techniques, adding new objects to an image is not typically considered a standard augmentation method as it can fundamentally change the content of the image."
        },
        {
            "question": "How does data augmentation help in reducing overfitting?",
            "options": [
                "By removing complex features from the data",
                "By increasing the amount and variety of training data",
                "By simplifying the model architecture",
                "By increasing the learning rate"
            ],
            "correct": "By increasing the amount and variety of training data",
            "explanation": "Data augmentation reduces overfitting by increasing the amount and variety of training data. This helps the model learn more generalized features rather than memorizing specific training examples."
        },
        {
            "question": "In the context of the MNIST dataset, which augmentation technique might NOT be appropriate?",
            "options": [
                "Rotation",
                "Horizontal flipping",
                "Adding random noise",
                "Slight translation"
            ],
            "correct": "Horizontal flipping",
            "explanation": "For the MNIST dataset (handwritten digits), horizontal flipping might not be appropriate as it can change the meaning of certain digits (e.g., 6 becomes 9). The other techniques are generally safe and can improve model robustness."
        }
    ]
    
    for i, q in enumerate(questions, 1):
        st.subheader(f"Question {i}")
        with st.container():
            st.write(q["question"])
            answer = st.radio("Select your answer:", q["options"], key=f"q{i}")
            if st.button("Check Answer", key=f"check{i}"):
                if answer == q["correct"]:
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Incorrect. The correct answer is: {q['correct']}")
                st.info(f"Explanation: {q['explanation']}")
            st.write("---")

if __name__ == "__main__":
    main()