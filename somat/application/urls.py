from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('', views.about, name="about"),
    path('', views.contact, name="contact"),
    path('search_subreddit/', views.search_subreddit, name='search_subreddit'),
    path('analyze_keyword/', views.analyze_keyword, name='analyze_keyword'),
]