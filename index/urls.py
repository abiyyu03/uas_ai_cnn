from django.urls import path
from . import views
urlpatterns = [
 path('', (views.create), name='create'),
 path('result/', (views.result), name='result')]