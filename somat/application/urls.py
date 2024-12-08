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
]