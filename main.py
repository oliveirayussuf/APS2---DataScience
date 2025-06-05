import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import numpy as np
import warnings

# Configurações iniciais
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_palette("pastel")

# Definir caminho do arquivo (ajuste conforme necessário)
file_path = 'data/Hotel_Reviews.csv'

# Carregar os dados
print("Carregando dados...")
try:
    # Ler o arquivo CSV diretamente para um DataFrame
    df = pd.read_csv(file_path, encoding='latin1')
    
    print(f"Dados carregados: {len(df)} registros")
    
    # Verificar colunas essenciais
    essential_cols = ['Reviewer_Score', 'Negative_Review', 'Positive_Review', 'Tags']
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        print(f"Erro: Colunas ausentes - {', '.join(missing_cols)}")
        exit()

    # Converter pontuação para numérico
    df['Reviewer_Score'] = pd.to_numeric(df['Reviewer_Score'], errors='coerce')
    
    # Remover linhas com pontuação ausente
    df = df.dropna(subset=['Reviewer_Score'])
    print(f"Registros após limpeza: {len(df)}")

except Exception as e:
    print(f"Erro ao carregar dados: {e}")
    exit()

# -------------------------------------------------------------------
# 1. ANÁLISE DE SENTIMENTO
# -------------------------------------------------------------------
print("\n[1/7] Analisando sentimentos...")
df['Review_Sentiment'] = np.where(df['Reviewer_Score'] > 5, 'Positive', 'Negative')
sentiment_counts = df['Review_Sentiment'].value_counts()

# -------------------------------------------------------------------
# 2. PROBLEMAS COMUNS
# -------------------------------------------------------------------
print("\n[2/7] Identificando problemas comuns...")
problem_keywords = [
    'construction', 'renovation', 'dirty', 'clean', 'noise', 
    'noisy', 'broken', 'smell', 'staff', 'service', 'breakfast',
    'maintenance', 'repair', 'issue', 'problem', 'faulty',
    'unclean', 'filthy', 'stain', 'damage', 'defect', 'malfunction',
    'poor', 'bad', 'horrible', 'terrible', 'awful', 'disappointing'
]

problem_counts = {}
for keyword in problem_keywords:
    count = df['Negative_Review'].astype(str).str.lower().str.contains(keyword).sum()
    problem_counts[keyword] = count

# Ordenar problemas mais comuns
sorted_problems = sorted(problem_counts.items(), key=lambda x: x[1], reverse=True)[:15]

# -------------------------------------------------------------------
# 3. TIPOS DE QUARTO
# -------------------------------------------------------------------
print("\n[3/7] Extraindo tipos de quarto...")
def extract_room_type(tags):
    try:
        matches = re.findall(r"'(.*?(?:Room|Suite|Studio|Apartment))'", str(tags), re.IGNORECASE)
        return matches[0] if matches else 'Unknown'
    except:
        return 'Unknown'

df['Room_Type'] = df['Tags'].apply(extract_room_type)
room_type_counts = df['Room_Type'].value_counts().head(10)

# -------------------------------------------------------------------
# 4. DURAÇÃO DA ESTADIA
# -------------------------------------------------------------------
print("\n[4/7] Calculando duração de estadias...")
def extract_stay_duration(tags):
    try:
        match = re.search(r"Stayed\s+(\d+)\s+(?:night|nights|noites|notti)", str(tags), re.IGNORECASE)
        return int(match.group(1)) if match else np.nan
    except:
        return np.nan

df['Stay_Duration'] = df['Tags'].apply(extract_stay_duration)
stay_duration_counts = df['Stay_Duration'].value_counts().sort_index().dropna()

# -------------------------------------------------------------------
# 5. ANÁLISE TEMPORAL
# -------------------------------------------------------------------
print("\n[5/7] Analisando tendências temporais...")
try:
    df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce', format='%m/%d/%Y')
    df['Year'] = df['Review_Date'].dt.year
    yearly_scores = df.groupby('Year')['Reviewer_Score'].mean().reset_index()
    yearly_counts = df['Year'].value_counts().sort_index().reset_index()
    yearly_counts.columns = ['Year', 'Count']
except Exception as e:
    print(f"Erro na análise temporal: {e}")
    yearly_scores = pd.DataFrame()
    yearly_counts = pd.DataFrame()

# -------------------------------------------------------------------
# 6. NACIONALIDADES (CORREÇÃO DO ERRO)
# -------------------------------------------------------------------
print("\n[6/7] Analisando por nacionalidade...")
# Converter explicitamente para string antes de usar .str
df['Reviewer_Nationality'] = df['Reviewer_Nationality'].astype(str).str.strip()
nationality_counts = df['Reviewer_Nationality'].value_counts().head(10)
nationality_scores = df.groupby('Reviewer_Nationality')['Reviewer_Score'].mean().loc[nationality_counts.index]

# -------------------------------------------------------------------
# 7. WORD CLOUDS
# -------------------------------------------------------------------
print("\n[7/7] Preparando nuvens de palavras...")
positive_text = ' '.join(df[df['Review_Sentiment'] == 'Positive']['Positive_Review'].astype(str))
negative_text = ' '.join(df[df['Review_Sentiment'] == 'Negative']['Negative_Review'].astype(str))

# ==================== VISUALIZAÇÕES ====================
print("\nGerando visualizações...")

# 1. Distribuição de Sentimentos
plt.figure(figsize=(10, 6))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', 
                     colors=['#66b3ff','#ff9999'],
                     explode=[0.05]*len(sentiment_counts),
                     shadow=True)
plt.title('Distribuição de Avaliações Positivas vs Negativas', fontsize=16)
plt.ylabel('')
plt.tight_layout()
plt.show()

# 2. Problemas Comuns
plt.figure(figsize=(14, 8))
problems, counts = zip(*sorted_problems)
sns.barplot(x=list(counts), y=list(problems), palette="Reds_r")
plt.title('Problemas Mais Frequentemente Citados', fontsize=16)
plt.xlabel('Quantidade de Menções')
plt.ylabel('Problema')
plt.tight_layout()
plt.show()

# 3. Nuvens de Palavras
def generate_wordcloud(text, title, color):
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap=color).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=15)
    plt.axis('off')
    plt.show()

generate_wordcloud(positive_text, 'Palavras Mais Comuns em Avaliações Positivas', 'Greens')
generate_wordcloud(negative_text, 'Palavras Mais Comuns em Avaliações Negativas', 'Reds')

# 4. Evolução Temporal
if not yearly_scores.empty and not yearly_counts.empty:
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(yearly_scores['Year'], yearly_scores['Reviewer_Score'], 'o-', color='#4c72b0')
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('Média de Pontuação', color='#4c72b0')
    ax1.tick_params(axis='y', labelcolor='#4c72b0')
    ax1.set_ylim(0, 10)
    
    ax2 = ax1.twinx()
    ax2.bar(yearly_counts['Year'], yearly_counts['Count'], color='#55a868', alpha=0.3)
    ax2.set_ylabel('Número de Avaliações', color='#55a868')
    ax2.tick_params(axis='y', labelcolor='#55a868')
    
    plt.title('Evolução das Avaliações ao Longo do Tempo', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 5. Tipos de Quarto
plt.figure(figsize=(14, 8))
sns.barplot(x=room_type_counts.values, y=room_type_counts.index, palette="Blues_d")
plt.title('Tipos de Quarto Mais Comuns', fontsize=16)
plt.xlabel('Quantidade')
plt.ylabel('Tipo de Quarto')
plt.tight_layout()
plt.show()

# 6. Duração da Estadia
plt.figure(figsize=(14, 7))
sns.barplot(x=stay_duration_counts.index.astype(str), 
            y=stay_duration_counts.values, 
            palette="rocket")
plt.title('Duração da Estadia (Noites)', fontsize=16)
plt.xlabel('Número de Noites')
plt.ylabel('Quantidade de Hospedagens')
plt.tight_layout()
plt.show()

# 7. Avaliações por Nacionalidade
plt.figure(figsize=(14, 10))
nationality_scores_sorted = nationality_scores.sort_values(ascending=True)
sns.barplot(x=nationality_scores_sorted.values, 
            y=nationality_scores_sorted.index, 
            palette="viridis")
plt.title('Média de Avaliações por Nacionalidade (Top 10)', fontsize=16)
plt.xlabel('Média de Pontuação')
plt.ylabel('Nacionalidade')
plt.xlim(0, 10)
plt.tight_layout()
plt.show()

# ==================== RESULTADOS NUMÉRICOS ====================
print("\n=== RESULTADOS DA ANÁLISE ===")
print(f"Total de avaliações: {len(df)}")
print(f"Avaliações positivas: {sentiment_counts.get('Positive', 0)} ({sentiment_counts.get('Positive', 0)/len(df)*100:.1f}%)")
print(f"Avaliações negativas: {sentiment_counts.get('Negative', 0)} ({sentiment_counts.get('Negative', 0)/len(df)*100:.1f}%)")

print("\nTop 10 problemas mais comuns:")
for i, (problem, count) in enumerate(sorted_problems[:10], 1):
    print(f"{i}. {problem.capitalize()}: {count} menções ({count/len(df)*100:.1f}%)")

print("\nTipos de quarto mais comuns:")
for i, (room_type, count) in enumerate(room_type_counts.items(), 1):
    print(f"{i}. {room_type}: {count} reservas ({count/len(df)*100:.1f}%)")

if not df['Stay_Duration'].dropna().empty:
    avg_stay = df['Stay_Duration'].mean()
    mode_stay = df['Stay_Duration'].mode()[0]
    print(f"\nDuração típica da estadia:")
    print(f"- Média: {avg_stay:.1f} noites")
    print(f"- Moda: {mode_stay} noites")

if not nationality_scores.empty:
    print("\nMelhores nacionalidades por avaliação:")
    for nat, score in nationality_scores.sort_values(ascending=False).head(5).items():
        print(f"- {nat}: {score:.1f}/10")
    
    print("\nPiores nacionalidades por avaliação:")
    for nat, score in nationality_scores.sort_values().head(5).items():
        print(f"- {nat}: {score:.1f}/10")

print("\nAnálise concluída com sucesso!")