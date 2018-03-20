# howdy/urls.py
from django.conf.urls import url
from howdy import views
from django.conf import settings
from django.conf.urls.static import static
#from django.conf import settings

urlpatterns = [
    url(r'^$', views.HomePageView.as_view(),name='home'),
    url(r'^about/$', views.AboutPageView.as_view()), # Add this /about/ route
    url(r'^photo/$', views.PhotoPageView.as_view()),
    url(r'^car/$', views.CarPageView.as_view()),
    url(r'^To-do/$', views.TodoPageView.as_view()),
    #url('django.views.static',(r'^media/(?P<path>.*)','serve',{'document_root':settings.MEDIA_ROOT}), ),
    url(r'^data/$', views.DataPageView.as_view()),  # Add this URL pattern
    #url(r'^static/(?P<path>.*)$', 'django.views.static.serve',{'document_root': settings.MEDIA_ROOT}),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
