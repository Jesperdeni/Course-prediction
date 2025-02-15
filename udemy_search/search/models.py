from django.db import models
from django.utils.timezone import now

class Course(models.Model):
    course_id = models.IntegerField(default=1000)  
    course_title = models.CharField(max_length=255)
    url = models.URLField(default="https://example.com")
    is_paid = models.BooleanField(default=True)
    price = models.CharField(max_length=20, default="Free") 
    num_subscribers = models.IntegerField(default=0)  
    num_reviews = models.IntegerField(default=0)  
    num_lectures = models.IntegerField(default=0)
    level = models.CharField(max_length=50, default="Beginner")
    content_duration = models.FloatField(default=0)
    published_timestamp = models.DateTimeField(default=now)  # Use default instead of auto_now_add
    subject = models.CharField(max_length=100, default="General")

    




    def __str__(self):
        return self.course_title
