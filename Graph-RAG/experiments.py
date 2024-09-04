import os

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq


NEO4J_URI = "neo4j+s://11605f97.databases.neo4j.io:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "xRlq2umwS_INLEj_lHMcL1cjQcC1b5gsDvqiMEc9RZU"

movie_query="""
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies.csv' as row

MERGE(m:Movie{id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') | 
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') | 
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') | 
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))

"""

graph=Neo4jGraph(url=NEO4J_URI,username=NEO4J_USERNAME,password=NEO4J_PASSWORD)
# graph.query(movie_query)
# graph.refresh_schema()

llm=ChatGroq(groq_api_key="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",model_name="llama-3.1-8b-instant")
chain=GraphCypherQAChain.from_llm(graph=graph,llm=llm,verbose=True)

print("Ask a question or press q to exit.....")
input_query = input("Enter your query: ")
while input_query != "q":
    if input_query == "" or input_query == " ":
        continue
    response=chain.invoke({"query":input_query})
    print(response)
    print("\nAsk another question or press q to exit.....")
    input_query = input("Enter your query: ")

print("Quiting.....")
