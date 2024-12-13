from django.test import TestCase
from views import get_image_from_wikidata, get_wikidata_id

# Create your tests here.

keyword_wikidata_id = get_wikidata_id("Donald Trump")
image_url = get_image_from_wikidata(keyword_wikidata_id)
print(f"Image URL: {image_url}")