import pandas as pd

# Define the languages, corresponding CSV files, and column names
languages = ["English", "German", "French", "Japanese"]
files = ['data/manual/english/english_pokemon_names_cleaned.csv', 'data/manual/german/german_pokemon_names_cleaned.csv', 'data/manual/french/french_pokemon_names_cleaned.csv', 'data/manual/japanese/japanese_pokemon_names_cleaned.csv']
column_names = ['Englisch', 'Deutsch', 'Französisch', 'Japanisch']

# Create an empty dictionary to hold the new data
new_pokemon_data = {"Names": {lang: [] for lang in languages}}

# Loop through the languages, corresponding CSV files, and column names
for lang, file, column_name in zip(languages, files, column_names):
    # Read the CSV file
    df = pd.read_csv(file)

    # Loop through the rows in the CSV file
    for index, row in df.iterrows():
        # Add the data to the dictionary in the desired format
        new_pokemon_data["Names"][lang].append(row[column_name])

print(new_pokemon_data)