from Conv_Network import multi_task_training, preprocess_images_only, evaluate_model
from sklearn.model_selection import train_test_split
import pandas as pd
import os

IMAGE_PATH = "Images"  
CSV_FILE = "styles.csv"
TEST_IMAGE = "15970.jpg"

def main():
    # ============================
    # CARREGAMENTO DO DATASET
    # ============================
    print("Lendo CSV...")
    df = pd.read_csv(CSV_FILE)

    # Selecionando colunas relevantes
    print("Selecionando as Colunas...")
    df = df.rename(columns={
        'articleType': 'category',
        'baseColour': 'color'
    })
    print("Atualizando a df...")
    
    # Filtrar classes com poucas instâncias
    min_class_count = 1
    class_counts = df['category'].value_counts()
    valid_classes = class_counts[class_counts >= min_class_count].index
    df = df[df['category'].isin(valid_classes)]
    df = df[['id', 'category', 'color']]

    # Remover linhas com valores ausentes
    df = df.dropna(subset=['category', 'color'])

    # ============================
    # SPLIT TREINO / TESTE
    # ============================
    print("Fazendo o split...")
    df = df.sample(n=5000, random_state=42) ##Para facilitar o teste
    df = df.groupby('category').filter(lambda x: len(x) > 1)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])

    # ============================
    # TREINAMENTO
    # ============================
    print("Iniciando treinamento...")
    model, category_encoder, color_encoder = multi_task_training(df_train)

    # ============================
    # PRÉ-PROCESSAMENTO DO TESTE
    # ============================
    print("Pré-processando dados de teste...")
    X_test_images, y_test_categories, y_test_colors = preprocess_images_only(df_test, category_encoder, color_encoder)

    # ============================
    # AVALIAÇÃO
    # ============================
    print("Avaliando modelo...")
    evaluate_model(model, X_test_images, y_test_categories, y_test_colors, category_encoder, color_encoder)

if __name__ == "__main__":
    main()