from django.conf.urls import url
from django.views.generic import TemplateView
from . import views

urlpatterns = [
    url(r'^$', views.ArtGenView.as_view(), name='index'),
    url(r'^history$', views.HistoryView.as_view(), name='history'),
    url(r'^historyParams', views.history_params, name='historyParams'),
    url(r'^customImage$', views.custom_image, name='customImage'),
]