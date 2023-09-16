# Open Information (Relation) Extraction Model
Build an open information (relation) extraction model making use of the algorithm introduced by TextRunner. This model can be used to get candidate relationships.

## Python Packages Needed
- pandas
- numpy
- pardata
- nltk
- spacy
- benepar (for training)
- networkx
- scikit-learn==1.0.2 (for training)

## Get Candidate Relationships
Train the model and save the objects as given in the code to a pickle object ("objects.pkl"). Load this object and run the get_candidate_relationships function to obtain candidate relationships in the form of triples. Triples are produced in the following format: (entity 1, candidate relationship, entity 2).