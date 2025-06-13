import requests
import time
from concurrent.futures import ThreadPoolExecutor

# Configurações
URL = "http://127.0.0.1:8000/predict"
IMAGE_PATH = "test.jpg"  # Caminho da imagem de teste (mesma usada para todas as requisições)

# Função para enviar uma requisição
def send_request():
    with open(IMAGE_PATH, "rb") as f:
        files = {'file': f}
        response = requests.post(URL, files=files)
        return response.status_code, response.elapsed.total_seconds()

# Escalabilidade Fraca: requisições sequenciais
def test_escalabilidade_fraca(num_requisicoes=50):
    print(f"\n Testando Escalabilidade Fraca: {num_requisicoes} requisições sequenciais")
    tempos = []
    for i in range(num_requisicoes):
        status, tempo = send_request()
        tempos.append(tempo)
        print(f"{i+1}/{num_requisicoes} - Status: {status} | Tempo: {tempo:.2f}s")
    print(f"Média de tempo por requisição: {sum(tempos)/len(tempos):.2f}s")

# Escalabilidade Forte: múltiplas requisições simultâneas
def test_escalabilidade_forte(num_requisicoes=50, max_workers=10):
    print(f"\n Testando Escalabilidade Forte: {num_requisicoes} requisições com {max_workers} clientes simultâneos")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        start = time.time()
        results = list(executor.map(lambda _: send_request(), range(num_requisicoes)))
        total_time = time.time() - start
        tempos = [r[1] for r in results]
    print(f"Tempo total: {total_time:.2f}s")
    print(f"Média de tempo por requisição: {sum(tempos)/len(tempos):.2f}s")
    print(f"Requisições com falha: {len([r for r in results if r[0] != 200])}")

# Executa ambos os testes
if __name__ == "__main__":
    test_escalabilidade_fraca(num_requisicoes=30)
    test_escalabilidade_forte(num_requisicoes=30, max_workers=10)
