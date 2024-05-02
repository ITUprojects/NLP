import pandas as pd

# Load the data
data = pd.read_csv(r'C:\Users\bogdan\Desktop\Uni\NLP\data_bogdan\Pokemon_names.txt', delimiter='\t')
# print(data.head())
# print(data.columns)
# print(data["Englisch"].head())

data.columns = data.columns.str.strip()

# Convert entire 'Deutsch' column to string and process to keep only alphabetical parts
data['Japanisch'] = data['Japanisch'].astype(str).apply(lambda x: ' '.join([item for item in x.split() if item.isalpha()]))

# Save the cleaned names to a CSV file
data[['Japanisch']].to_csv('japanese_pokemon_names_cleaned.csv', index=False)

print("Cleaned Pokémon names have been saved to 'japanese_pokemon_names_cleaned.csv'.")