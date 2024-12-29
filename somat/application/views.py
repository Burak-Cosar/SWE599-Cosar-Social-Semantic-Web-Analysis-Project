import requests, os, spacy, json, time, re
from django.shortcuts import render
from django.http import JsonResponse
from datetime import datetime, timezone
from dotenv import load_dotenv
from collections import defaultdict
from flair.models import SequenceTagger
from flair.data import Sentence
from rapidfuzz import process, fuzz
from urllib.parse import quote

load_dotenv()
tagger = SequenceTagger.load("ner-fast")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Create your views here.
def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

# Reddit API Authentication
token_cache = {
    'access_token': None,
    'expires_at': 0
}

def get_reddit_token():
    global token_cache
    if token_cache['access_token'] and datetime.now().timestamp() < token_cache['expires_at']:
        return token_cache['access_token']

    CLIENT_ID = os.getenv('CLIENT_ID')
    SECRET_ID = os.getenv('SECRET_ID')
    USERNAME = os.getenv('USERNAME')
    PASSWORD = os.getenv('PASSWORD')

    if not all([CLIENT_ID, SECRET_ID, USERNAME, PASSWORD]):
        raise Exception('Missing Reddit API credentials.')

    auth = requests.auth.HTTPBasicAuth(CLIENT_ID, SECRET_ID)
    data = {'grant_type': 'password', 'username': USERNAME, 'password': PASSWORD}
    headers = {'User-Agent': 'MyAPI/0.0.1'}

    response = requests.post(
        'https://www.reddit.com/api/v1/access_token',
        auth=auth, data=data, headers=headers
    )

    if response.status_code != 200:
        raise Exception(f"Authentication failed: {response.json().get('error', 'Unknown error')}")

    token_data = response.json()
    token_cache['access_token'] = token_data['access_token']
    token_cache['expires_at'] = datetime.now().timestamp() + token_data['expires_in'] - 60
    return token_cache['access_token']

def search_subreddit(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    query = request.GET.get('query', '').strip()
    if not query:
        return JsonResponse({'error': 'Query is required.'}, status=400)

    try:
        token = get_reddit_token()
        headers = {'Authorization': f'bearer {token}', 'User-Agent': 'MyAPI/0.0.1'}
        response = requests.get(
            'https://oauth.reddit.com/subreddits/search',
            headers=headers, params={'q': query, 'limit': 10}
        )
        if response.status_code != 200:
            return JsonResponse({'error': 'Failed to fetch subreddits.'}, status=response.status_code)
        subreddits = response.json().get('data', {}).get('children', [])
        results = [{'name': sub['data']['display_name'], 'subscribers': sub['data']['subscribers']} for sub in subreddits]
        return JsonResponse({'results': results})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
def get_reddit_data(request):
    subreddit = request.GET.get('subreddit', 'all')
    keyword = request.GET.get('keyword', '').strip()
    time_period = request.GET.get('time_period', 'all')

    token = get_reddit_token()
    headers = {'Authorization': f'bearer {token}', 'User-Agent': 'MyAPI/0.0.1'}

    results = []
    after = None
    while len(results) < 1000:
        params = {
            'q': keyword,
            'limit': 100,
            'sort': 'new',
            't': time_period,
            'after': after,
        }
        if subreddit != 'all':
            url = f'https://oauth.reddit.com/r/{subreddit}/search'
            params['restrict_sr'] = True
        else:
            url = 'https://oauth.reddit.com/search'

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            break

        data = response.json().get('data', {})
        posts = data.get('children', [])
        after = data.get('after')

        if not posts:
            print("No more posts available.")
            break

        for post in posts:
            post_data = post['data']
            created_date = datetime.fromtimestamp(
                post_data.get('created_utc', 0), tz=timezone.utc
            ).strftime('%Y-%m-%d %H:%M:%S')
            results.append({
                'title': post_data.get('title'),
                'body': post_data.get('selftext', post_data.get('body', '')),
                'subreddit': post_data.get('subreddit'),
                'author': post_data.get('author'),
                'score': post_data.get('score'),
                'date': created_date,
            })

        if not after:
            print("Pagination complete.")
            break

        time.sleep(1)

    # Save results to reddit_data.json
    with open('reddit_data.json', 'w') as f:
        json.dump({'results': results}, f, indent=4)
        
    return JsonResponse({'results': results})

def normalize_quotes(text):
    # Replace curly single and double quotes with their standard counterparts
    text = re.sub(r'[‘’]', "'", text)  # Replace curly single quotes
    text = re.sub(r'[“”]', '"', text)  # Replace curly double quotes
    return text

def extract_named_entities(posts):
    # Extract named entities using Flair NER model.
    entity_counts = defaultdict(lambda: {"label": None, "count": 0})

    for post in posts:
        # Combine title and body
        text = f"{post.get('title', '')} {post.get('body', '')}".strip()
        text = normalize_quotes(text)  # Normalize quotes in the text

        if not text:
            continue  # Skip empty posts

        # Create a Flair Sentence object
        sentence = Sentence(text)

        # Predict named entities
        tagger.predict(sentence)

        # Count entities
        for entity in sentence.get_spans("ner"):
            entity_text = normalize_quotes(entity.text)  # Normalize entity text
            entity_label = entity.tag

            if entity_text not in entity_counts:
                entity_counts[entity_text]["label"] = entity_label
            entity_counts[entity_text]["count"] += 1

    return entity_counts

def analyze_keyword(request):
    keyword = request.GET.get('keyword', '').strip()
    query = request.GET.get('query', '').strip()

    reddit_response = get_reddit_data(request)
    reddit_data = json.loads(reddit_response.content).get("results", [])

    if not reddit_data:
        return JsonResponse({'error': 'No data found from Reddit API.'}, status=404)

    # Extract named entities
    entity_counts = extract_named_entities(reddit_data)

    # Prepare entities for linking
    analysis_results = [
        {"entity": entity, "label": data["label"], "count": data["count"]}
        for entity, data in entity_counts.items()
    ]

    # Perform entity linking
    keyword_entity, linked_entities = spacy_entity_linking(analysis_results, keyword, query)

    if not linked_entities:
        return JsonResponse({'error': 'No linked entities found.'}, status=404)

    # Categorize and sort entities by label
    categorized_entities = defaultdict(list)
    for entity in linked_entities:
        categorized_entities[entity['label']].append(entity)

    # Sort entities within each category and fetch images only for top 5
    linked_people = sorted(
        categorized_entities['PER'], key=lambda x: x['count'], reverse=True
    )[:5]

    linked_locations = sorted(
        categorized_entities['LOC'], key=lambda x: x['count'], reverse=True
    )[:5]

    linked_organizations = sorted(
        categorized_entities['ORG'], key=lambda x: x['count'], reverse=True
    )[:5]

    # Fetch images for top entities only
    for entity_list in [linked_people, linked_locations, linked_organizations]:
        for entity in entity_list:
            entity['wikidata_id'] = get_wikidata_id(entity['entity'], query)
            entity['image'] = get_image_from_wikidata(entity['wikidata_id'])
            entity['wikipedia'] = get_wikipedia_url(entity['wikidata_id'])

    keyword_entity_wikidata_id = get_wikidata_id(keyword, query)
    keyword_entity_image = get_image_from_wikidata(keyword_entity_wikidata_id)
    keyword_entity_wikipedia = get_wikipedia_url(keyword_entity_wikidata_id)

    # Construct the output JSON
    output_data = {
        "keyword_entity": {
            "entity": keyword_entity['entity'],
            "label": keyword_entity['label'],
            "count": keyword_entity['count'],
            "wikidata_id": keyword_entity_wikidata_id,
            "image": keyword_entity_image,
            "wikipedia": keyword_entity_wikipedia
        },
        "linked_people": linked_people,
        "linked_locations": linked_locations,
        "linked_organizations": linked_organizations
    }

    # Save the output to a JSON file
    with open('output_data.json', 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    # Return the JSON response
    return JsonResponse(output_data)

ENTITY_ALIAS_MAPPING = {
    "Trump": ("Donald Trump", "PER"),
    "Musk": ("Elon Musk", "PER"),
    "Hegseth": ("Pete Hegseth", "PER"),
    "U.S.": ("United States", "LOC"),
    "US": ("United States", "LOC"),
    "Biden": ("Joe Biden", "PER"),
    "America": ("United States", "LOC"),
    "DeSantis": ("Ron DeSantis", "PER"),
    "Kamala": ("Kamala Harris", "PER"),
    "Harris": ("Kamala Harris", "PER"),
    "Zelenskyy": ("Volodymyr Zelenskyy", "PER"),
    "Zelensky": ("Volodymyr Zelenskyy", "PER"),
    "Putin": ("Vladimir Putin", "PER"),
    "FOX": ("Fox News", "ORG"),
    "Dries": ("Dries Mertens", "PER"),
    "Artificial Intelligence": ("AI", "MISC"),
    "AI": ("AI", "MISC"),
    "Taylor Swift": ("Taylor Swift", "PER"),
    "SZA": ("SZA", "PER"),
    "Tyla,": ("Tyla", "PER"),
    "Gala": ("Galatasaray", "ORG"),
    "Tadic": ("Dusan Tadic", "PER"),
    "Guler": ("Arda Guler", "PER"),
    "Güler": ("Arda Guler", "PER"),
    "Bellingham": ("Jude Bellingham", "PER"),
    "Adams'": ("Adams", "PER"),
    "Zuckerberg": ("Mark Zuckerberg", "PER"),
    "Bezos": ("Jeff Bezos", "PER"),
    "Altman": ("Sam Altman", "PER"),
    "Ramaswamy": ("Vivek Ramaswamy", "PER"),
    "LA": ("Los Angeles", "LOC"),
    "Salah": ("Mohamed Salah", "PER"),
    "Messi": ("Lionel Messi", "PER"),
    "Mbappé": ("Kylian Mbappé", "PER"),
    "Man United": ("Manchester United", "ORG"),
    "Fetterman": ("John Fetterman", "PER"),
    "Witcher": ("The Witcher", "ORG"),
    "Baldur": ("Baldur's Gate", "ORG"),
    "Baldurs Gate": ("Baldur's Gate", "ORG"),
    "Kendricks": ("Kendrick Lamar", "PER"),
    "Tyler": ("Tyler, the Creator", "PER"),
    "Smiths": ("The Smiths", "ORG"),
    "smiths": ("The Smiths", "ORG"),
    "Kanye": ("Kanye West", "PER"),
    "The Smiths": ("The Smiths", "ORG"),
    "Spotify": ("Spotify", "ORG"),
    "Mbappe": ("Kylian Mbappé", "PER"),
    "Trafford": ("Old Trafford", "LOC"),
    "Manchester United Edition": ("Manchester United", "ORG"),
    "Germany": ("Germany", "LOC"),
    "Guardiola": ("Pep Guardiola", "PER"),
    "Al": ("Al Ahly", "ORG"),
    "Ross County": ("Ross County", "LOC"),
    "Luis Diaz": ("Luis Díaz", "PER"),
}

def spacy_entity_linking(entities, keyword, query, threshold=85):
    merged_entities = defaultdict(lambda: {"label": None, "count": 0})
    keyword_entity = None
    query = query.lower()

    for entity_data in entities:
        entity = entity_data['entity']
        label = entity_data['label']
        count = entity_data['count']

        # Split compound entities and process individually
        entity_parts = re.split(r'[:\-\/]', entity)
        for part in entity_parts:
            resolved_entity = part.strip()

            # Apply alias mapping
            if resolved_entity in ENTITY_ALIAS_MAPPING:
                resolved_entity, resolved_label = ENTITY_ALIAS_MAPPING[resolved_entity]
                label = resolved_label
            else:
                # Use SpaCy NER as fallback
                doc = nlp(resolved_entity)
                if doc.ents:
                    resolved_entity = doc.ents[0].text

            # Fuzzy match with existing merged entities
            match_data = process.extractOne(
                resolved_entity, merged_entities.keys(), scorer=fuzz.token_set_ratio
            )
            if match_data:
                match, score, _ = match_data
                if score >= threshold:
                    merged_entities[match]["count"] += count
                    continue

            # Add or update the entity in the merged dictionary
            merged_entities[resolved_entity]["label"] = label
            merged_entities[resolved_entity]["count"] += count

    # Extract the keyword entity
    for key, data in list(merged_entities.items()):
        if keyword.lower() in key.lower():
            keyword_entity = {"entity": key, "label": data["label"], "count": data["count"]}
            del merged_entities[key]
            break

    # Prepare related entities list
    related_entities = [
        {"entity": k, "label": v["label"], "count": v["count"]}
        for k, v in merged_entities.items()
    ]

    return keyword_entity, related_entities


def get_wikidata_id(search_term, context):
    """
    Fetch the Wikidata ID for a given search term, optionally filter results locally using context.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": search_term,
        "language": "en",
        "format": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data.get('search'):
        # Retrieve all results
        results = data['search']
        
        # Try filtering results locally based on context if provided
        if context:
            filtered_results = [
                item for item in results
                if context in item.get('description', '').lower()
            ]
            if filtered_results:
                return filtered_results[0]['id']  # Return the first matching result

        # Return the first result if no match is found or context is not provided
        return results[0]['id']
    else:
        return "Not found"
    
def get_image_from_wikidata(wikidata_id):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "props": "claims",
        "format": "json"
    }
    response = requests.get(url, params=params).json()

    if 'entities' in response and wikidata_id in response['entities']:
        claims = response['entities'][wikidata_id].get('claims', {})

        # Check for P154 (logo) first
        if 'P154' in claims:
            image_data = claims['P154'][0].get('mainsnak', {}).get('datavalue', {})
            if image_data:
                image_file_name = image_data.get('value')
                image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(image_file_name.replace(' ', '_'))}"
                return image_url

        # Check for P18 (image)
        if 'P18' in claims:
            image_data = claims['P18'][0].get('mainsnak', {}).get('datavalue', {})
            if image_data:
                image_file_name = image_data.get('value')
                image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(image_file_name.replace(' ', '_'))}"
                return image_url

    # Return None if no image is found
    return "None"

def get_wikipedia_url(wikidata_id):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "props": "sitelinks",
        "format": "json",
        "sitefilter": "enwiki",  # Filter for English Wikipedia
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Handle HTTP errors
        data = response.json()

        # Extract the English Wikipedia title
        sitelinks = data.get("entities", {}).get(wikidata_id, {}).get("sitelinks", {})
        if "enwiki" in sitelinks:
            title = sitelinks["enwiki"].get("title")
            if title:
                # Construct the URL using the title
                return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        return None  # Return None if no title is found
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Wikidata: {e}")
        return None
    

# KNOWLEDGE GRAPH GENERATION
from rdflib import Graph, URIRef, Literal, Namespace, RDF
from rdflib.namespace import FOAF, RDFS, SKOS
from SPARQLWrapper import SPARQLWrapper, JSON

# Define namespaces
WD = Namespace("http://www.wikidata.org/entity/")
SOMAT = Namespace("http://localhost:8000/somat/")

def get_entity_data(wikidata_id, delay=0.8):
    if not wikidata_id or wikidata_id == "Not found":
        return {}

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT ?property ?propertyLabel ?value ?valueLabel ?instanceOfLabel
    WHERE {{
        OPTIONAL {{
            wd:{wikidata_id} ?prop ?value .
            ?property wikibase:directClaim ?prop .
        }}
        OPTIONAL {{
            wd:{wikidata_id} wdt:P31 ?instanceOf .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
    }}
    LIMIT 100
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    print(f"Waiting {delay} seconds before querying {wikidata_id}...")
    time.sleep(delay)

    results = sparql.query().convert()
    enriched_data = {}

    for result in results["results"]["bindings"]:
        if "propertyLabel" in result and "valueLabel" in result:
            property_label = result["propertyLabel"]["value"]
            value_label = result["valueLabel"]["value"]
            if "T" in value_label:
                value_label = value_label.split("T")[0]
            enriched_data[property_label] = value_label

        if "instanceOfLabel" in result:
            enriched_data["instance of"] = result["instanceOfLabel"]["value"]

    allowed_properties = {"date of birth", "spouse", "occupation", "place of birth", "award received", "instance of"}
    return {k: v for k, v in enriched_data.items() if k in allowed_properties}

def create_enriched_graph(data):
    g = Graph()

    # Bind the SKOS namespace
    g.bind('skos', SKOS)

    keyword_entity = data.get("keyword_entity")
    if not keyword_entity or not keyword_entity.get("wikidata_id"):
        print("Error: Central keyword entity is missing or invalid.")
        return None

    # Add the central keyword node
    keyword_node = URIRef(WD[keyword_entity["wikidata_id"]])
    g.add((keyword_node, RDF.type, SOMAT.Entity))
    g.add((keyword_node, RDFS.label, Literal(keyword_entity["entity"])))

    # Enrich the central node
    enriched_data = get_entity_data(keyword_entity["wikidata_id"])
    for prop, value in enriched_data.items():
        prop_node = URIRef(SOMAT[prop.replace(" ", "_")])
        value_node = Literal(value)
        g.add((keyword_node, prop_node, value_node))

    # Define property URIs
    SOMAT_count = URIRef("http://localhost:8000/somat/count")
    SOMAT_instance_of = URIRef("http://localhost:8000/somat/instance_of")

    def add_linked_entities(entities, relationship):
        # Create a dictionary to track entities by their Wikidata ID
        entity_map = {}

        for entity in entities:
            if not entity.get("wikidata_id") or entity["wikidata_id"] == "Not found":
                print(f"Skipping entity with invalid wikidata_id: {entity}")
                continue

            wikidata_id = entity["wikidata_id"]

            # If we've seen this entity before, update its count
            if wikidata_id in entity_map:
                entity_map[wikidata_id]["total_count"] += entity["count"]
                # Store alternative labels
                if entity["entity"] not in entity_map[wikidata_id]["labels"]:
                    entity_map[wikidata_id]["labels"].append(entity["entity"])
            else:
                entity_map[wikidata_id] = {
                    "total_count": entity["count"],
                    "primary_label": entity["entity"],
                    "labels": [entity["entity"]]
                }

        # Now create nodes for each unique entity
        for wikidata_id, info in entity_map.items():
            related_node = URIRef(WD[wikidata_id])

            # Add type
            g.add((related_node, RDF.type, SOMAT.Entity))

            # Add primary label only once
            g.add((related_node, RDFS.label, Literal(info["primary_label"])))

            # Add alternative labels using SKOS
            for alt_label in info["labels"][1:]:  # Skip primary label
                g.add((related_node, SKOS.altLabel, Literal(alt_label)))

            # Add single combined count
            g.add((related_node, SOMAT_count, Literal(info["total_count"])))

            # Add instance_of property and human-specific predicates
            enriched_related_data = get_entity_data(wikidata_id)
            instance_of_info = enriched_related_data.get("instance of")
            if instance_of_info:
                g.add((related_node, SOMAT_instance_of, Literal(instance_of_info)))

            human_specific_features = {"date of birth", "occupation", "place of birth", "spouse"}
            for feature in human_specific_features:
                if feature in enriched_related_data:
                    feature_uri = URIRef(SOMAT[feature.replace(" ", "_")])
                    g.add((related_node, feature_uri, Literal(enriched_related_data[feature])))

            # Link to central entity
            g.add((keyword_node, relationship, related_node))

    # Add people, locations, and organizations
    add_linked_entities(data.get("linked_people", []), SOMAT.mentioned_with)
    add_linked_entities(data.get("linked_locations", []), SOMAT.mentioned_in_location)
    add_linked_entities(data.get("linked_organizations", []), SOMAT.associated_with)

    # Serialize the graph to an RDF/XML file
    g.serialize("enriched_graph.rdf", format="xml")

    return g

def generate_knowledge_graph(request):
    with open('output_data.json', 'r') as f:
        data = json.load(f)

    knowledge_graph = create_enriched_graph(data)

    graph_data = knowledge_graph.serialize(format="json-ld")

    return JsonResponse(json.loads(graph_data), safe=False)