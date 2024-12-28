import requests
from django.http import JsonResponse
from .views import get_wikidata_id, get_image_from_wikidata

# Create your tests here.

def wikidata_id_lookup(request):
    search_term = request.GET.get('search_term', '').strip()
    context = request.GET.get('context', '').strip()

    if not search_term:
        return JsonResponse({'error': 'search_term is required.'}, status=400)

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
        results = data['search']
        
        # Filter results based on context
        if context:
            filtered_results = [
                item for item in results
                if context.lower() in item.get('description', '').lower()
            ]
            if filtered_results:
                return JsonResponse({'id': filtered_results[0]['id'], 'description': filtered_results[0]['description']})
        
        # Return the first result if no match with context
        return JsonResponse({'id': results[0]['id'], 'description': results[0].get('description', 'No description')})
    else:
        return JsonResponse({'id': 'Not found', 'description': 'No matching entities found.'})