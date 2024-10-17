# search/urls.py
from django.urls import path
from .views import search_view

urlpatterns = [
    path('', search_view, name='search_view'),
]
