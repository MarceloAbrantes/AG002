# pip install pandas
# pip install scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option('future.no_silent_downcasting', True)

#Leitura de dados (primeira parte)
# Read the csv file
df = pd.read_csv("wholesale.csv")

print('Lendo os dados que estão chegando')
print(df.head())



#Substituir valores nas colunas usando o método replace (segunda parte)
print('\n Substituir os valores da Coluna usando o metodo replace:')
df['Channel'] = df['Channel'].replace({'HoReCa': 0, 'Retail': 1}).astype(int)
df['Region'] = df['Region'].replace({'Lisbon': 0, 'Oporto': 1, 'Other': 2}).astype(int)


print(df.head())


print('\n Colunas reordenadas (reindex):')
#Reordenar colunas usando reindex e o atributo columns (terceira parte)
nova_ordem_colunas = ['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen', 'Channel']

df = df.reindex(columns=nova_ordem_colunas)
print(df.head())
print('\n',df.columns) #Print somente das colunas

# Separar os dados em X (features) e y (target)
X = df.drop('Channel', axis=1)  # Todas as colunas exceto 'Channel' são as características
y = df['Channel']  # 'Channel' é o alvo

# Separar os dados em 80% treinamento e 20% teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n Lendo treinamento")
print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)

#Árvore de decisão
#Treinar a Árvore de Decisão
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

#Fazer previsões no conjunto de teste
y_pred = classifier.predict(X_test)

# Avaliar precisão
accuracy = accuracy_score(y_test, y_pred)
print(f'Acuracia da Arvore de Decisao: ,{accuracy * 100:.2f}%')

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Função para capturar dados do usuário e fazer a previsão
def classificar_novo_dado(classifier):
    print("Insira os valores para cada uma das características:")
    
    # Capturar os valores das features do usuário
    region = int(input("Region (0 para Lisbon, 1 para Oporto, 2 para Other): "))
    fresh = float(input("Fresh: "))
    milk = float(input("Milk: "))
    grocery = float(input("Grocery: "))
    frozen = float(input("Frozen: "))
    detergents_paper = float(input("Detergents_Paper: "))
    delicatessen = float(input("Delicatessen: "))

    # Criar um DataFrame temporário com os dados do usuário
    novo_dado = pd.DataFrame({
        'Region': [region],
        'Fresh': [fresh],
        'Milk': [milk],
        'Grocery': [grocery],
        'Frozen': [frozen],
        'Detergents_Paper': [detergents_paper],
        'Delicatessen': [delicatessen]
    })
    
    # Fazer a previsão usando o modelo treinado
    predicao = classifier.predict(novo_dado)[0]
    
    # Converter o resultado da previsão para o nome do canal
    canal = "HoReCa" if predicao == 0 else "Retail"
    print(f"\nO canal de vendas previsto para os dados inseridos é: {canal}")

# Chamando a função após o treinamento do modelo
classificar_novo_dado(classifier)

