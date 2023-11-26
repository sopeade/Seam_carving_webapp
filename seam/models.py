from django.db import models

# Create your models here.
class Image(models.Model):
    cover = models.ImageField(upload_to='input/')


