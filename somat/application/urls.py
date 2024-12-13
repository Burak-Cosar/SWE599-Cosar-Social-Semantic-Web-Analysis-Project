from django.urls import path
from . import views

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
]