from django.test import TestCase
from .models import Course

class CourseModelTest(TestCase):
    def test_course_creation(self):
        course = Course.objects.create(title="Python Basics", subject="Programming", num_subscribers=1000, price=49.99)
        self.assertEqual(course.title, "Python Basics")
