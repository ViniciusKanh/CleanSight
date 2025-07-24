
# ✨ CleanSight

🔍 **CleanSight** é uma plataforma interativa de pré-processamento de dados que transforma arquivos brutos em datasets prontos para análise e modelagem de machine learning. Com uma interface moderna e endpoints robustos, ela permite upload, análise, limpeza e visualização de dados diretamente no navegador.

---

## 📦 Funcionalidades

- 📁 **Upload inteligente de datasets** (CSV, TXT, TSV)
- 🔍 **Análise automática** de encoding, delimitadores e tipos de dados
- 🧹 **Tratamento de valores ausentes**
- 🧬 **Codificação de variáveis categóricas e booleanas**
- 🧮 **Normalização de variáveis numéricas**
- 🧠 **Seleção de features mais relevantes**
- ⚖️ **Balanceamento de classes**
- 🧊 **Remoção de duplicatas**
- 📈 **Gráficos de PCA, correlação e outliers**
- 📝 **Perfilamento completo do dataset** (pandas profiling)
- 💾 **Download do dataset processado**
- 🌐 **API RESTful** com múltiplos endpoints úteis

---

## 🛠️ Tecnologias Utilizadas

| Camada     | Tecnologias                 |
|------------|-----------------------------|
| Backend 🧠 | Python, Flask               |
| Frontend 💻 | HTML5, CSS3, JavaScript     |
| Estilo 🎨  | Tailwind-like custom CSS     |
| Outros 🔧  | Matplotlib, NumPy, chardet   |

---

## 🚀 Como Executar Localmente

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/cleansight.git
cd cleansight
```

### 2. Crie um ambiente virtual e instale as dependências
```bash
python -m venv venv
source venv/bin/activate  # ou `venv\Scripts\activate` no Windows
pip install -r requirements.txt
```

> 📦 Certifique-se de que o arquivo `requirements.txt` contenha:
```txt
Flask
chardet
matplotlib
numpy
```

### 3. Execute o servidor Flask
```bash
python main.py
```

### 4. Acesse no navegador 🌐
```
http://localhost:5000
```

---

## 🎯 Endpoints Disponíveis

| Método | Rota             | Função                                 |
|--------|------------------|----------------------------------------|
| POST   | `/api/upload`    | Faz upload do dataset                  |
| POST   | `/api/analyze`   | Analisa a coluna target e qualidade    |
| POST   | `/api/process`   | Realiza o pré-processamento completo   |
| POST   | `/api/statistics`| Estatísticas e perfilamento do dataset |
| POST   | `/api/pca`       | Gera gráfico PCA 2D                    |
| POST   | `/api/outliers`  | Detecta e visualiza outliers           |
| GET    | `/api/download`  | Baixa o dataset processado             |
| POST   | `/api/clear`     | Limpa arquivos e estado da sessão      |

Para ver um relatório detalhado com o **pandas profiling**, utilize a rota
`/api/statistics` ou clique em **Perfil do Dataset** na interface.

---

## 📸 Exemplos Visuais

![Exemplo de Upload e Análise](docs/example_upload.png)
![Gráfico PCA](docs/example_pca.png)
![Distribuição das Classes](docs/example_target.png)

---

## 🧠 Sugestões de Uso

- 💡 Ideal para cientistas de dados em fase de exploração de dados
- 🎓 Perfeito para projetos acadêmicos com foco em aprendizado de máquina
- 🏢 Útil para empresas que precisam preparar dados antes da modelagem

---

## 👤 Autor

**Vinicius Santos**  
Engenheiro da Computação • Cientista de Dados 

[🔗 LinkedIn](https://www.linkedin.com/in/vinicius-santos)  
[🐙 GitHub](https://github.com/ViniciusKanh)

---

## 📃 Licença

Este projeto está licenciado sob a **MIT License**.  
Sinta-se livre para usar, modificar e compartilhar. 🚀
