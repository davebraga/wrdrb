import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from keras.applications import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam

IMAGE_PATH = "Images"  
CSV_FILE = "styles.csv"
INPUT_SHAPE = (224, 224, 3)

#===============================================================================================
#================================ PRÉ-PROCESSAMENTO ============================================

def preprocessing(df): 
    # Inicializando os encoders
    category_encoder = LabelEncoder()
    color_encoder = LabelEncoder()

    df = df.sample(n=750, random_state=42)

    # Criar o caminho completo do arquivo
    df["image_file"] = df["id"].astype(str) + ".jpg"
    df["full_path"] = df["image_file"].apply(lambda x: os.path.join(IMAGE_PATH, x))

    # Filtrar apenas arquivos que realmente existem
    df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

    # Fit nos encoders
    df["category_encoded"] = category_encoder.fit_transform(df["category"])
    df["color_encoded"] = color_encoder.fit_transform(df["color"].str.lower().str.strip())

    # Inicializando os arrays
    X_images = []
    y_categories = []
    y_colors = []

    for idx, row in df.iterrows():
        img = cv2.imread(row["full_path"])
        if img is None: 
            print(f"Imagem não encontrada: {row['full_path']}") 
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) / 255.0  # normalizando [0,1]
        X_images.append(img)
        y_categories.append(row["category_encoded"])
        y_colors.append(row["color_encoded"])

    # Convertendo para numpy
    X_images = np.array(X_images)
    y_categories = np.array(y_categories)
    y_colors = np.array(y_colors)
    
    print("DATA SHAPES")
    print(f"X_images: {X_images.shape}")
    print(f"y_categories: {y_categories.shape}")
    print(f"y_colors: {y_colors.shape}")

    return X_images, y_categories, y_colors, category_encoder, color_encoder

#===============================================================================================
#=========================================== MODELO ============================================

def multi_task_training(df):

    print("1) Pré-Processamento!")
    # PRE-PROCESSAMENTO
    X_images, y_categories, y_colors, category_encoder, color_encoder = preprocessing(df) 

    print("2) Construindo modelo!")
    # CONSTRUÇÃO DO MODELO
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    # Saídas multi-task
    category_output = Dense(len(category_encoder.classes_), activation="softmax", name="category_output")(x)
    color_output = Dense(len(color_encoder.classes_), activation="softmax", name="color_output")(x)

    model = Model(inputs=base_model.input, outputs=[category_output, color_output])
    model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={
        "category_output": "sparse_categorical_crossentropy",
        "color_output": "sparse_categorical_crossentropy",
    },
    metrics={
        "category_output": "accuracy",
        "color_output": "accuracy"
    }
)

    print("3) Fazendo Treinamento!")
    # TREINAMENTO
    history = model.fit(
        X_images,
        {"category_output": y_categories, "color_output": y_colors},
        epochs=5,
        batch_size=32,
        validation_split=0.2
    )

    return model, category_encoder, color_encoder

#===============================================================================================
#========================================== PREDIÇÃO ===========================================   

def prediction(model, category_encoder, color_encoder, image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    category_pred, color_pred = model.predict(img)

    category_idx = np.argmax(category_pred, axis=1)[0]
    color_idx = np.argmax(color_pred, axis=1)[0]

    category_label = category_encoder.inverse_transform([category_idx])[0]
    color_label = color_encoder.inverse_transform([color_idx])[0]

    result_category = "CATEGORY: " + category_label
    result_color = "COLOR: " + color_label

    return result_category, result_color

#===============================================================================================
#====================================== AVALIAÇÃO DO MODELO ====================================

def evaluate_model(model, X_test, y_test_categories, y_test_colors, category_encoder, color_encoder):
    if len(X_test) == 0:
        print("⚠️ Nenhuma amostra para avaliar!")
        return

    predictions = model.predict(X_test)
    category_pred, color_pred = predictions

    if len(y_test_categories) != len(category_pred):
        print(f"⚠️ Inconsistência: {len(y_test_categories)} labels vs {len(category_pred)} predictions")
        return

    category_pred_labels = category_encoder.inverse_transform(np.argmax(category_pred, axis=1))
    color_pred_labels = color_encoder.inverse_transform(np.argmax(color_pred, axis=1))

    category_true_labels = category_encoder.inverse_transform(y_test_categories)
    color_true_labels = color_encoder.inverse_transform(y_test_colors)

    print("Category Classification Report:")
    print(classification_report(category_true_labels, category_pred_labels))

    category_cm = confusion_matrix(category_true_labels, category_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(category_cm, annot=True, fmt='d', cmap='Blues', xticklabels=category_encoder.classes_, yticklabels=category_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Category Confusion Matrix')
    plt.show()

    print("Color Classification Report:")
    print(classification_report(color_true_labels, color_pred_labels))

    color_cm = confusion_matrix(color_true_labels, color_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(color_cm, annot=True, fmt='d', cmap='Blues', xticklabels=color_encoder.classes_, yticklabels=color_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Color Confusion Matrix')
    plt.show()


def preprocess_images_only(df_subset, category_encoder, color_encoder):
    X_images = []
    y_categories = []
    y_colors = []

    for idx, row in df_subset.iterrows():
        image_path = os.path.join(IMAGE_PATH, f"{row['id']}.jpg")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Imagem não encontrada: {image_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) / 255.0
        X_images.append(img)

        valid_color = row["color"].lower().strip()
        valid_category = row["category"]

        if valid_color in color_encoder.classes_ and valid_category in category_encoder.classes_:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Imagem não encontrada: {image_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) / 255.0
            X_images.append(img)

            y_colors.append(color_encoder.transform([valid_color])[0])
            y_categories.append(category_encoder.transform([valid_category])[0])
        else:
            print(f"Pulando {row['id']} — classe desconhecida: color={valid_color}, category={valid_category}")
            continue

    X_images = np.array(X_images)
    y_categories = np.array(y_categories)
    y_colors = np.array(y_colors)

    print(f"Pré-processamento final: {X_images.shape[0]} imagens válidas")
    return X_images, y_categories, y_colors

