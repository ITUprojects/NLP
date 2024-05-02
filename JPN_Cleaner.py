""" Finds and removes from the dataset rows with ONLY non-Japanese characters in the Japanese synopsis. """

import pandas as pd
import re

# Load the CSV file
df = pd.read_csv('pokemon_synopses.csv')

original = df.shape[0]

# Find any row with ONLY non-Japanese characters
# '[^\u30A0-\u30FF]+' is the regex for non-Japanese characters
non_japanese = df[df['overview_jpn'].str.contains('[^\u30A0-\u30FF]+')]

# Drop the rows with non-Japanese characters that also don't have Japanese characters
# '[\u3040-\u30FF]+' is the regex for Japanese characters
non_japanese = non_japanese[~non_japanese['overview_jpn'].str.contains('[\u3040-\u30FF]+')]
df = df.drop(non_japanese.index)

# Save the cleaned CSV file
df.to_csv('pokemon_synopses_cleaned.csv', index=False)

# print removed rows
print('Removed:', non_japanese.shape[0])
print('Row text:', non_japanese['overview_jpn'].values)

# print size of original and cleaned data
print('Original:', original)
print('Cleaned:', df.shape[0])