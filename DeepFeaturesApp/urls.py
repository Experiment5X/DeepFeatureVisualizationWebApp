from django.conf.urls import url
from django.views.generic import TemplateView
from . import views

urlpatterns = [
    url(r'^$', views.ArtGenView.as_view(), name='index'),
    url(r'customImage', views.custom_image, name='customImage'),
]