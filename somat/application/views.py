import requests, os, spacy, json
from django.shortcuts import render
from django.http import JsonResponse
from datetime import datetime, timezone
from dotenv import load_dotenv
from collections import defaultdict
from flair.models import SequenceTagger
from flair.data import Sentence
from collections import defaultdict
from rapidfuzz import process, fuzz
import spacy

load_dotenv()
tagger = SequenceTagger.load("ner-fast")

# Create your views here.
def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def search_subreddit(request):
    if request.method == 'GET':
        # Get the query parameter
        query = request.GET.get('query', '').strip()
        if not query:
            return JsonResponse({'error': 'Query is required.'}, status=400)

        print(f"Received query: {query}")  # Debug log

        # Retrieve Reddit API credentials from environment variables
        CLIENT_ID = os.getenv('CLIENT_ID')
        SECRET_ID = os.getenv('SECRET_ID')
        USERNAME = os.getenv('USERNAME')
        PASSWORD = os.getenv('PASSWORD')

        # Validate the presence of critical environment variables
        if not all([CLIENT_ID, SECRET_ID, USERNAME, PASSWORD]):
            return JsonResponse(
                {'error': 'Missing Reddit API credentials in environment variables.'},
                status=500,
            )

        # Reddit API Authentication
        auth = requests.auth.HTTPBasicAuth(CLIENT_ID, SECRET_ID)
        data = {
            'grant_type': 'password',
            'username': USERNAME,
            'password': PASSWORD,
        }
        headers = {'User-Agent': 'MyAPI/0.0.1'}

        # Request access token
        token_response = requests.post(
            'https://www.reddit.com/api/v1/access_token',
            auth=auth,
            data=data,
            headers=headers,
        )

        # Check if token request was successful
        if token_response.status_code != 200:
            print("Token Request Failed:", token_response.json())  # Debug log
            return JsonResponse(
                {'error': f"Authentication failed: {token_response.json().get('error', 'Unknown error')}"},
                status=token_response.status_code,
            )

        # Extract access token
        TOKEN = token_response.json().get('access_token')
        if not TOKEN:
            return JsonResponse({'error': 'Failed to retrieve access token.'}, status=403)

        # Search for subreddits
        headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}
        response = requests.get(
            'https://oauth.reddit.com/subreddits/search',
            headers=headers,
            params={'q': query, 'limit': 10},
        )

        # Check if subreddit search was successful
        if response.status_code != 200:
            print("Subreddit Search Failed:", response.json())  # Debug log
            return JsonResponse(
                {'error': f"Reddit API error: {response.json().get('message', 'Unknown error')}"},
                status=response.status_code,
            )

        # Extract subreddit data
        subreddits = response.json().get('data', {}).get('children', [])
        if not subreddits:
            return JsonResponse({'error': 'No subreddits found for the given query.'}, status=404)

        # Format the results
        results = [
            {
                'name': sub['data']['display_name'],
                'subscribers': sub['data']['subscribers'],
            }
            for sub in subreddits
        ]
        return JsonResponse({'results': results})

    # If method is not GET, return a 405 Method Not Allowed response
    return JsonResponse({'error': 'Method not allowed.'}, status=405)

def save_api_data_to_file(data, filename='api_data.json'):
    # Save API data to a JSON file.
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

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
    
def get_reddit_data(request):
    subreddit = request.GET.get('subreddit', 'all')
    keyword = request.GET.get('keyword', '').strip()

    CLIENT_ID = os.getenv('CLIENT_ID')
    SECRET_ID = os.getenv('SECRET_ID')
    USERNAME = os.getenv('USERNAME')
    PASSWORD = os.getenv('PASSWORD')

    if not all([CLIENT_ID, SECRET_ID, USERNAME, PASSWORD]):
        return JsonResponse({'error': 'Missing Reddit API credentials.'}, status=500)

    # Authenticate with Reddit API
    auth = requests.auth.HTTPBasicAuth(CLIENT_ID, SECRET_ID)
    data = {'grant_type': 'password', 'username': USERNAME, 'password': PASSWORD}
    headers = {'User-Agent': 'MyRedditApp/1.0'}

    token_response = requests.post(
        'https://www.reddit.com/api/v1/access_token',
        auth=auth,
        data=data,
        headers=headers,
    )
    if token_response.status_code != 200:
        return JsonResponse({'error': 'Failed to authenticate with Reddit API.'}, status=403)

    TOKEN = token_response.json().get('access_token')
    if not TOKEN:
        return JsonResponse({'error': 'Failed to retrieve access token.'}, status=403)

    headers['Authorization'] = f"bearer {TOKEN}"

    # Fetch posts and comments using Reddit API
    results = []
    after = None
    while len(results) < 1000:
        params = {
            'q': keyword,
            'limit': 100,
            'sort': 'new',
            't': 'all',
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

        data = response.json()
        posts = data.get('data', {}).get('children', [])
        for post in posts:
            post_data = post['data']
            created_date = datetime.fromtimestamp(
                post_data.get('created_utc', 0),
                tz=timezone.utc
            ).strftime('%Y-%m-%d %H:%M:%S')
            results.append({
                'title': post_data.get('title'),
                'body': post_data.get('selftext', post_data.get('body', '')),
                'subreddit': post_data.get('subreddit'),
                'author': post_data.get('author'),
                'score': post_data.get('score'),
                'date': created_date,
            })

        after = data.get('data', {}).get('after')
        if not after:
            break  # No more pages

    # Save results to a JSON file
    save_api_data_to_file(results, filename='reddit_data.json')

    return JsonResponse({'results': results})

def analyze_keyword(request):
    subreddit = request.GET.get('subreddit', 'all')
    keyword = request.GET.get('keyword', '').strip()

    reddit_response = get_reddit_data(request)
    reddit_data = json.loads(reddit_response.content).get("results", [])

    if not reddit_data:
        return JsonResponse({'error': 'No data found from Reddit API.'}, status=404)

    entity_counts = extract_named_entities(reddit_data)
    sorted_entities = sorted(
        entity_counts.items(),
        key=lambda item: item[1]["count"],
        reverse=True
    )[:50]

    analysis_results = [
        {"entity": entity, "label": data["label"], "count": data["count"]}
        for entity, data in sorted_entities
    ]

    # Perform entity linking
    keyword_entity, linked_entities = spacy_entity_linking(analysis_results, keyword)

    # Categorize by label
    linked_people = [
        entity for entity in linked_entities if entity['label'] == 'PER'
    ][:5]

    linked_locations = [
        entity for entity in linked_entities if entity['label'] == 'LOC'
    ][:5]

    linked_organizations = [
        entity for entity in linked_entities if entity['label'] == 'ORG'
    ][:5]

    return JsonResponse({
        "keyword_entity": keyword_entity,
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

    # Prepare remaining entities list
    remaining_entities = [
        {'entity': k, 'label': v['label'], 'count': v['count']}
        for k, v in merged_entities.items()
    ]

    return keyword_entity, remaining_entities