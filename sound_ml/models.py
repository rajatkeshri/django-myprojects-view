from django.db import models

# Create your models here.
class audiofile(models.Model):
    file = models.FileField(upload_to = "files")
