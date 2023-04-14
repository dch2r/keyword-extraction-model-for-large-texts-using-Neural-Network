
import json
import pandas as pd
import networkx as nx
import os
from flask import Flask
from flask import render_template, request, jsonify
from webapp import app
from collections import Counter
import math

#app = Flask(__name__)
# load data
graph=nx.read_gpickle("webapp/Data/graph.gpickle")

IMG = os.path.join('static', 'Images')
app.config['UPLOAD_FOLDER'] = IMG

with open("webapp/Data/tags.txt","r") as f:
    t2=f.read()


def tag_dict(text):
    '''
    description : convert raw tags per row from json to integer corrosponding each tags from the whole file.
    
    Argument : string text
    
    return : dictionary of tags to integer dic
    '''
    
    #preprocess text
    text=text.lower()
    text=text.split('\n')
    dic={}
    
    #store each tag as a unique integer to represent
    for i,v in enumerate(text):
        v=v.split(',')
        dic[i]=v
        del dic[i][len(dic[i])-1]
    return dic

def tag_to_int(data):
    '''
    description : word to integer method that returns dictionary of word to int used for enique tags to int.
    
    Argument : string text
    
    return : dictionary of tag to int
    '''
    word_int={}
    for i,v in enumerate(data):
        word_int[v]=i
    return word_int

def unique_tags(text):
    '''
    description : get tags and return unique list of tags.
    
    Argument : string text
    
    return : list tags
    '''
    text=text.split('\n')
    tags=list(set(''.join(text).split(',')))
    tags=sorted(tags)
    del tags[0]
    return tags

def int_to_tags(int_tag):
    '''
    description : input integer representation of tag and return its string form.
    
    Argument : int tags
    
    return : dictionary of int o tags
    '''
    return {v:k for k,v in list(int_tag.items())}

tag_unique=unique_tags(t2)
int_tag=tag_to_int(tag_unique)
tags_dictionary=tag_dict(t2)
int_to_tag=int_to_tags(int_tag)

def query_fun(search_term,relations,topk=0.3):
    '''
    description : input search term, graph of tags and topk or the top confidence or related terms to get 
                    as output.
    
    Argument : string search term, onject graph, int topk
    
    return : list (predicted words and int weights)
    '''
    #convert search term string to int.

    try:
        current=int_tag[search_term]
    
    except:
        return " "
    # Search for neighbours of search term
    
    similar=list(relations.neighbors(current))
    
    nodes_similar=[]
    for i in similar:
        nodes_similar.append((i,relations.get_edge_data(current,i)['weight']))
    
    #sort tags according to their edges weight, highest weight is first placed
    nodes_similar.sort(key=lambda x:x[1],reverse=True)
    
    nodes_similar_descent=[]
    #Extrack topk words from sorted of similar words for prediction of search term
    for i in range(math.ceil(topk*len(nodes_similar))):
        
        nodes_similar_descent.append((nodes_similar[i][0],nodes_similar[i][1]))
    predict_terms=[]
    
    for i in nodes_similar_descent:
        predict_terms.append((int_to_tag[i[0]],i[1]))
    
    return predict_terms

@app.route('/')
@app.route('/index')
def show_index():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'network.jpg')
    return render_template("master.html", user_image = full_filename)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    query=str(query)
    query=query.lower()
    if " " in query:
        query=query.replace(" ","-")

    # use model to predict classification for query
    results=query_fun(query,graph)
    if results == " ":
        results=["No result found, Try adding "-" between the word"]

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        recommend=results
    )

#if __name__ == '__main__':
#    app.run(port="5001",host="0.0.0.0",debug=True)
