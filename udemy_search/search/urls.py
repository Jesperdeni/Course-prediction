# search/urls.py
from django.urls import path
from .views import search_courses  # Import the view function

urlpatterns = [
    path('', search_courses, name='search_courses'),  # Ensure this is mapped correctly
]
