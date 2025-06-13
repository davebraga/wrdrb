# 👕 Classificador de Roupas - Categoria e Cor

Este projeto implementa um sistema inteligente baseado em **Redes Neurais Convolucionais (CNN)** para classificação automática de imagens de roupas. A aplicação é composta por um front-end web (React) e um back-end em Python com FastAPI, integrando um modelo treinado com TensorFlow/Keras.

---

## 📌 Funcionalidades

- Upload de imagem de roupa via interface web.
- Classificação automática da **categoria** (camiseta, calça, vestido etc.).
- Predição da **cor predominante** da peça.
- API REST para uso externo e integração.
- Testes de escalabilidade (forte e fraca) com múltiplas requisições.

---

## 🚀 Tecnologias Utilizadas

### 🖼️ Front-end
- [React](https://reactjs.org/)
- [TypeScript](https://www.typescriptlang.org/)
- [shadcn/ui](https://ui.shadcn.com/)
- Axios / Fetch API

### 🧠 Back-end
- [Python 3.10+](https://www.python.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [TensorFlow 2.15 / Keras](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [OpenCV](https://opencv.org/)
- [Pillow (PIL)](https://python-pillow.org/)
- [Uvicorn](https://www.uvicorn.org/) – ASGI server

### 📦 Testes & Ferramentas
- [ngrok](https://ngrok.com/) – Túnel para testes públicos
- [`requests`](https://docs.python-requests.org/)
- [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html)

### 📁 Estrutura de Arquivos

├── app.py                 # API FastAPI (ponto de entrada)
├── trained_model.keras   # Modelo CNN treinado
├── category_encoder.pkl  # Encoder da categoria
├── color_encoder.pkl     # Encoder da cor
├── requirements.txt      # Dependências do projeto
├── frontend/             # Interface React (opcional)
└── scalability_test.py   # Script para testes de carga

### 🤝 Autores

- Davi Lorenzo B. Braga

- Felipe Augusto Morais
