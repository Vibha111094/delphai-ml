Method 1: Simple entity linking based on entity-mention match: 
- Perform NER using a pre-trained spacy pipeline.
- Filter out the ORG entities as we are interested in companies.
- Search for the ORG names in the company_collection name list and match them with the corresponding URL.

Implementation in simple_entity_linking.ipynb, results in news_articles-linked.jsonl

This method although simple, works in almost 60% of the cases, however, the con is that it does not take into account the description of the companies, and how that relates to the context in which the entity is used.

Method 2: Semantic Entity Linking

This approach typically has the following steps:  
Text ---> NER ----> Candidate Generation ----->Entity Linking

This can be done in different ways:

1. Using Knowledge Bases and training a Spacy Entity linking Pipeline Model
- Create a KB by adding the entities 
{url: vector_embedding(industry_label+headquarter+company_description)} 
also the alias
{url: company_name}
- Then create a training data set in the spacy training format using the gold dataset
- Add entity linker factory to the NER pipeline.
- Train the entity linker using the training data set that we created.
- This would create a pipeline that can link each mention to its corresponding entity.

Partial implementation in train_entity_linking.ipynb

Faced challenges with the spacy version incompatibility issue, and linking the Knowledge base to the spacy pipeline.

2. Transformer Based Entity Linking  
Create 2 encoders:
- Context encoder
- Candidate encoder

Input to the Context encoder would be Context, Mention, Context.  
Input to the Candidate encoder would be Title, Description.  
The Siamese network is trained such that the if the candidate refers to the mention in the context the score is 1, or else it is 0.

Once the network is trained, the Nearest neighbor algorithm is used to generate candidate entities, followed by the entity linking.
![Alt text](<Screenshot 2024-01-23 at 9.12.23 PM.png>)









