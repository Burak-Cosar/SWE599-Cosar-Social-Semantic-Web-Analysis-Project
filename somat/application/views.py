import requests, os, spacy, json, time
from django.shortcuts import render
from django.http import JsonResponse
from datetime import datetime, timezone
from dotenv import load_dotenv
from collections import defaultdict
from flair.models import SequenceTagger
from flair.data import Sentence
from collections import defaultdict
from rapidfuzz import process, fuzz

load_dotenv()
tagger = SequenceTagger.load("ner-fast")

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
            params['restrict_sr'] = True  # Restrict search to the subreddit
        else:
            url = 'https://oauth.reddit.com/search'

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            break  # Stop if there's an error

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

        time.sleep(1)  # Delay to prevent rate limiting

    # Save results to reddit_data.json
    with open('reddit_data.json', 'w') as f:
        json.dump({'results': results}, f, indent=4)
        
    return JsonResponse({'results': results})


def extract_named_entities(posts):
    
    # Extract named entities using Flair NER model.
    
    entity_counts = defaultdict(lambda: {"label": None, "count": 0})

    for post in posts:
        # Combine title and body
        text = f"{post.get('title', '')} {post.get('body', '')}".strip()

        if not text:
            continue  # Skip empty posts

        # Create a Flair Sentence object
        sentence = Sentence(text)

        # Predict named entities
        tagger.predict(sentence)

        # Count entities
        for entity in sentence.get_spans("ner"):
            entity_text = entity.text
            entity_label = entity.tag

            if entity_text not in entity_counts:
                entity_counts[entity_text]["label"] = entity_label
            entity_counts[entity_text]["count"] += 1

    return entity_counts

def analyze_keyword(request):
    subreddit = request.GET.get('subreddit', 'all')
    keyword = request.GET.get('keyword', '').strip()

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
    keyword_entity, linked_entities = spacy_entity_linking(analysis_results, keyword)

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
            wikidata_id = get_wikidata_id(entity['entity'])
            entity['image'] = get_image_from_wikidata(wikidata_id)

    keyword_wikidata_id = get_wikidata_id(keyword_entity['entity'])
    keyword_entity_image = get_image_from_wikidata(keyword_wikidata_id)

    return JsonResponse({
        "keyword_entity": {
            "entity": keyword_entity['entity'],
            "label": keyword_entity['label'],
            "count": keyword_entity['count'],
            "image": keyword_entity_image
        },
        "linked_people": linked_people,
        "linked_locations": linked_locations,
        "linked_organizations": linked_organizations,
    })

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

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
    ", Drake": ("Drake", "PER"),
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
}

from collections import defaultdict
from rapidfuzz import process, fuzz
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def spacy_entity_linking(entities, keyword, threshold=85):
    merged_entities = defaultdict(lambda: {"label": None, "count": 0})
    keyword_entity = None

    for entity_data in entities:
        entity = entity_data['entity']
        label = entity_data['label']
        count = entity_data['count']

        # Apply manual alias mapping
        if entity in ENTITY_ALIAS_MAPPING:
            resolved_entity, resolved_label = ENTITY_ALIAS_MAPPING[entity]
            label = resolved_label
        else:
            resolved_entity = entity

        # Use SpaCy NER if needed
        doc = nlp(resolved_entity)
        if doc.ents:
            resolved_entity = doc.ents[0].text

        # Use fuzzy matching to merge similar entities
        match_data = process.extractOne(
            resolved_entity, merged_entities.keys(), scorer=fuzz.token_set_ratio
        )

        if match_data:
            match, score, _ = match_data
            if score >= threshold:
                merged_entities[match]["count"] += count
                continue

        # Add or update the entity in merged_entities
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
        {'entity': k, 'label': v['label'], 'count': v['count']}
        for k, v in merged_entities.items()
    ]

    return keyword_entity, related_entities

def get_wikidata_id(search_term):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": search_term,
        "language": "en",
        "format": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data['search']:
        return data['search'][0]['id']
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
        claims = response['entities'][wikidata_id]['claims']
        if 'P18' in claims:
            image_file_name = claims['P18'][0]['mainsnak']['datavalue']['value']
            image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{image_file_name.replace(' ', '_')}"
            return image_url
    return "None"