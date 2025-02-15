
# udemy_search/urls.py
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('search/', include('search.urls')),  # Ensure this line is correct
    path('', RedirectView.as_view(url='/search/', permanent=True)),  # Redirect root URL
]

