web: gunicorn seam_carving_webapp.wsgi:application
worker: celery -A seam_carving_webapp worker -l info
heroku ps:scale celeryworker=1