# Classificação de Pneumonia em Raios-X Torácicos utilizando CNNs 
## Definição do Problema e Dataset
A pneumonia é uma infecção pulmonar que afeta milhões de pessoas anualmente, especialmente crianças e idosos. O diagnóstico preciso por meio de raios-X torácicos é crucial para o tratamento adequado. Este projeto visa desenvolver um sistema de apoio diagnóstico automatizado utilizando Redes Neurais Convolucionais (CNNs) para classificar imagens de raios-X como "Normal" ou "Pneumonia".

Para isso, utilizaremos o dataset [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) disponível no Kaggle. Nossos experimentos foram compostos por:
- 5.856 imagens de raios-X torácicos (JPEG)

- 2 categorias: NORMAL (1.583 imagens) e PNEUMONIA (4,273 imagens)

- A classe PNEUMONIA está subdividida em bacteriana e viral, mas não trabalhamos com essa subclassificação

## Orientações para executar o projeto
Para configurar e executar este projeto, siga os passos abaixo:

### 1\. Preparação do Ambiente

  * **Clone o Repositório:** Comece clonando este repositório para sua máquina local.
    ```bash
    git clone [LINK_DO_REPOSITORIO]
    cd [NOME_DO_REPOSITORIO]
    ```
  * **Versão do Python:** Recomenda-se usar o **Python 3.11**. É altamente aconselhável criar e ativar um ambiente virtual para isolar as dependências do projeto:
    ```bash
    python3.11 -m venv .venv
    .\.venv\Scripts\activate # Para ativar no Windows:
    # Para ativar no macOS/Linux:
    # source .venv/bin/activate
    ```
  * **Instale as Dependências:** Com o ambiente virtual ativado, instale todas as bibliotecas necessárias:
    ```bash
    pip install -r requirements.txt
    ```

### 2\. Estrutura dos Dados

Certifique-se de que seus dados de imagem estão organizados no diretório `data/` com a seguinte estrutura:

```
data/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3\. Execução do Projeto

  * **Modelos:** As implementações dos modelos de CNN estão no arquivo `models/main.py`.
  * **Experimentos:** Os experimentos e a análise são realizados nos notebooks (`.ipynb`) localizados no diretório `experiments/`.

Para executar os experimentos, abra e execute os notebooks na sequência desejada.