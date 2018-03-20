# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

# Create your views here.

 # howdy/views.py
from django.shortcuts import render
from django.views.generic import TemplateView

# Create your views here.
class HomePageView(TemplateView):
    '''def get(self, request, **kwargs):
        return render(request, 'index.html', context=None)'''
    template_name="index.html"

# Add this view
class AboutPageView(TemplateView):
    template_name = "about.html"

# Photo view
class PhotoPageView(TemplateView):
    template_name = "photo.html"

# Car view
class CarPageView(TemplateView):
    template_name = "car.html"

#To-do view
class TodoPageView(TemplateView):
    template_name="To-do.html"

class DataPageView(TemplateView):
    def get(self, request, **kwargs):
        # we will pass this context object into the
        # template so that we can access the data
        # list in the template
        context = {
            'data': [
                {
                    'name': 'Vaibhav',
                    'worth': '1000000000000'
                }
            ]
        }

        return render(request, 'data.html', context)

