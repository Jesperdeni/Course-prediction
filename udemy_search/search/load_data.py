import csv
from datetime import datetime
from search.models import Course

CSV_FILE_PATH = "C:\\Users\\DENI\\Desktop\\pr\\udemy_search\\data\\udemy_courses.csv"  # Update this path

def load_csv_data():
    with open(CSV_FILE_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            course, created = Course.objects.get_or_create(
                course_id=row["course_id"],
                defaults={
                    "course_title": row["course_title"],
                    "url": row["url"],
                    "is_paid": row["is_paid"].lower() == "true",
                    "price": row["price"],
                    "num_subscribers": int(row["num_subscribers"]),
                    "num_reviews": int(row["num_reviews"]),
                    "num_lectures": int(row["num_lectures"]),
                    "level": row["level"],
                    "content_duration": float(row["content_duration"]),
                    "published_timestamp": datetime.strptime(row["published_timestamp"], "%Y-%m-%dT%H:%M:%SZ"),
                    "subject": row["subject"],
                }
            )
            
            if created:
                print(f"Added course: {row['course_title']}")
            else:
                print(f"Course {row['course_id']} already exists.")

if __name__ == "__main__":
    load_csv_data()
