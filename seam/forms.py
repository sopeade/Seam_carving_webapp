from django import forms
from .models import Image


class PostForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ["cover"]
        labels = {"cover": ""}
