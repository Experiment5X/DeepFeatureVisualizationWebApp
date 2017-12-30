from django.shortcuts import redirect
from django.http import HttpResponse
from django.views.generic import FormView, TemplateView
from django.shortcuts import render
from .forms import DeepFeatureForm
from . import feature_creations
from .layer_info import prepare_histories
import os
import cv2
import json


worker_thread_created = False


# Create your views here.
class ArtGenView(FormView):
    template_name = 'DeepFeaturesApp/index.html'
    form_class = DeepFeatureForm

    def get(self, request):
        form = DeepFeatureForm()
        history = []
        if 'history' in request.session:
            history = request.session['history']

        layers_selected = [False] * 22
        layers_selected[1] = True
        return render(request, 'DeepFeaturesApp/index.html', {'form': form,
                                                              'history': prepare_histories(history),
                                                              'layers_selected': layers_selected})

    def post(self, request):
        form = DeepFeatureForm(request.POST)

        if form.is_valid():
            # to prevent the form from rendering with green outlines for good submission
            new_form = DeepFeatureForm()
            new_form.fields['epoch_count'].initial = form.data['epoch_count']
            new_form.fields['grad_std_clip'].initial = form.data['grad_std_clip']
            new_form.fields['image_std_clip'].initial = form.data['image_std_clip']
            new_form.fields['layer_index'].initial = form.data['layer_index']
            new_form.fields['learning_rate'].initial = form.data['learning_rate']
            new_form.fields['total_variation'].initial = form.data['total_variation']
        else:
            new_form = form

        if 'custom_image' in request.session:
            original_image_path = request.session['custom_image']
        else:
            original_image_path = './DeepFeaturesApp/static/DeepFeaturesApp/pineapple.jpg'
        start_image = cv2.resize(cv2.imread(original_image_path), (224, 224))

        filename = feature_creations.generate_image_name()
        params = feature_creations.ImageParameters(start_image, filename, form.cleaned_data['learning_rate'],
                                                   form.cleaned_data['layer_index'], form.cleaned_data['image_std_clip'],
                                                   form.cleaned_data['grad_std_clip'], form.cleaned_data['epoch_count'],
                                                   form.cleaned_data['total_variation'])

        layer_index = int(form.cleaned_data['layer_index'])
        layers_selected = [False] * 22 # one for each layer
        layers_selected[layer_index - 1] = True

        history = {
            'filename': filename[17:],
            'original': original_image_path[17:],
            'params': form.data
        }

        if 'history' not in request.session:
            request.session['history'] = []

        request.session['history'].append(history)
        request.session.modified = True

        global worker_thread_created
        if not worker_thread_created:
            worker_thread_created = True

            worker = feature_creations.AsyncImageFeatureCreator()
            worker.start()

        feature_creations.images_to_make.put(params)

        image_name = os.path.basename(filename)
        return render(request, 'DeepFeaturesApp/index.html', {'form': new_form,
                                                              'image_path': image_name,
                                                              'history':
                                                                  prepare_histories(request.session['history'][:-1]),
                                                              'layers_selected': layers_selected})


def write_image(f):
    file_name = feature_creations.generate_image_name()
    with open(file_name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return file_name


def custom_image(request):
    if request.method == 'POST':
        request.session['custom_image'] = write_image(request.FILES['custom-image'])
        return redirect('/')


def history_params(request):
    if request.method == 'GET':
        history_index = int(request.GET.get('index', '0'))
        hist_record = request.session['history'][history_index]

        hist_json = json.dumps(hist_record['params'])
        return HttpResponse(hist_json)


class HistoryView(TemplateView):
    template_name = 'DeepFeaturesApp/history.html'

    def get(self, request):
        history = []
        if 'history' in request.session:
            history = request.session['history']

        return render(request, 'DeepFeaturesApp/history.html', { 'history': prepare_histories(history, all=True)})