from django.urls import path
from . import views
from . import tests

urlpatterns = [
    path('', views.home, name="home"),
    path('', views.about, name="about"),
    path('', views.contact, name="contact"),
    path('search_subreddit/', views.search_subreddit, name='search_subreddit'),
    path('get_reddit_data/', views.get_reddit_data, name='get_reddit_data'),
    path('analyze_keyword/', views.analyze_keyword, name='analyze_keyword'),
    path('spacy_entity_linking/', views.spacy_entity_linking, name='spacy_entity_linking'),
    path('get_reddit_token/', views.get_reddit_token, name='get_reddit_token'),
    path('get_wikidata_id/', views.get_wikidata_id, name='get_wikidata_id'),
    path('get_image_from_wikidata/', views.get_image_from_wikidata, name='get_image_from_wikidata'),
    path('wikidata_id_lookup/', tests.wikidata_id_lookup, name='wikidata_id_lookup'),
    path('get_entity_data/', views.get_entity_data, name='get_entity_data'),
    path('generate_knowledge_graph/', views.generate_knowledge_graph, name='generate_knowledge_graph'),
]