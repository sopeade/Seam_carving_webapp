
from django.urls import path
from seam import views

urlpatterns = [
    # path('', views.index.as_view(), name='index'),
    # path('', views.home, name='home'),
    path('', views.index, name='index'),
    path('seam_carving', views.seam_carving, name='seam_carving'),
    path('download_file', views.download_file, name='download_file'),
    path('video_file', views.video_file, name='video_file'),
]
