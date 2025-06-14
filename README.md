 Detecção de Violência em Vídeos

Este projeto utiliza um modelo de deep learning para classificar a presença de violência em imagens e vídeos. A arquitetura combina uma CNN (MobileNetV2) com um LSTM Bidirecional para análise de características espaciais e temporais.

---

### Pré-requisitos
* Python (versões 3.8 a 3.11)

---

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Crie e ative um ambiente virtual (opcional, mas recomendado):**
    ```bash
    python3 -m venv venv

    # No Windows
    venv/Scripts/activate
    # No macOS/Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências a partir do `requirements.txt`:**
```bash
pip install -r requirements.txt
```


### Como Usar

#### Treinamento do Modelo
Para treinar o modelo, organize os vídeos de treinamento nas pastas `data/Violence` e `data/NonViolence`. Em seguida, execute o notebook `violence_train.ipynb`. Os modelos gerados serão salvos na pasta `models/`.

#### Previsão em Imagens
Para classificar uma única imagem, utilize o script `violence_predict.py`.

**Comando:**
```bash
python violence_predict.py <path/frame.png>
#Exemplo
python violence_predict.py imgs_test_not-violence/5.jpg
```

##### Resultado do teste ######

Ao executar a previsão, a imagem de entrada será salva na pasta imgs_results com um rotulo indicando a classificação ("Violence" ou "Non-Violence") e o percentual de confiança.





