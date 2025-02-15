from django.contrib import admin
from .models import Course

@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ('course_title', 'price', 'num_subscribers', 'published_timestamp')
    search_fields = ('course_title', 'subject')

