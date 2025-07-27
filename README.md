# Classificação de Flores com Redes Neurais Convolucionais (CNNs)

## Definição do Problema e Dataset

A classificação de imagens é um desafio fundamental em visão computacional, com aplicações diversas, desde sistemas de reconhecimento de objetos até catalogação botânica. Este projeto tem como objetivo desenvolver um sistema de classificação automatizado utilizando **Redes Neurais Convolucionais (CNNs)** para identificar diferentes espécies de flores a partir de imagens.

Para isso, utilizaremos o dataset **[Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)** disponível no Kaggle. Nossos experimentos são baseados nas seguintes características do dataset:

  * **Tamanho:** 4.317 imagens de flores (JPEG).
  * **Categorias:** 5 classes distintas de flores:
      * `daisy` (margarida) com 764 imagens
      * `dandelion` (dente-de-leão) com 1052 imagens
      * `rose` (rosa) com 784 imagens
      * `sunflower` (girassol) com 733 imagens
      * `tulip` (tulipa) com 984 imagens

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
├── daisy/
├── dandelion/
├── rose/
├── sunflower/
└── tulip/
```

### 3\. Execução do Projeto

  * **Modelos:** As implementações dos modelos de CNN estão no arquivo `models/main.py`.
  * **Experimentos:** Os experimentos e a análise são realizados nos notebooks (`.ipynb`) localizados no diretório `experiments/`.

Para executar os experimentos, abra e execute os notebooks na sequência desejada.