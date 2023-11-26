commands: python manage.py runserver
celery-local: celery -A seam_carving_webapp worker -l info
redis-local: redis-server