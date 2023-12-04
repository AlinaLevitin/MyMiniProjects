import pandas as pd


file = 'folding-nicer-format.data'

with open(file) as f:
    contents = f.read()

delimiter = '<\n<\n<\n<\n<\n<\n<\n<\n<\n<\n<\n<\n'

unparsed_proteins = contents.split(delimiter)

proteins = [unparsed_protein.split('\n') for unparsed_protein in unparsed_proteins]

amino_acids = []
for protein in proteins:
  for amino_acid in protein:
    amino_acid = amino_acid.split(' ')
    amino_acids.append(amino_acid)

df = pd.DataFrame(amino_acids, columns =['amino_acid', 'secondary_structure'])
df.dropna(inplace=True)

df.drop(index=21744, inplace=True)

name = 'amino_acids_secondary_structure.csv'

df.to_csv(name, index=False)
