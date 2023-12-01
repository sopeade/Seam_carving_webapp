
from django.urls import path
from seam import views

urlpatterns = [
    path('', views.index, name='index'),
    path('seam_carving', views.seam_carving, name='seam_carving'),
    path('download_file', views.download_image, name='download_image'),
    path('download_video', views.download_video, name='download_video'),
]
