from django.shortcuts import redirect
from django.http import HttpResponse
from django.views.generic import FormView
from django.shortcuts import render
from .forms import DeepFeatureForm
from . import feature_creations
import os
import cv2


worker_thread_created = False


# Create your views here.
class ArtGenView(FormView):
    template_name = 'DeepFeaturesApp/index.html'
    form_class = DeepFeatureForm

    def get(self, request):
        form = DeepFeatureForm()
        return render(request, 'DeepFeaturesApp/index.html', {'form': form})

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
            start_image = cv2.resize(cv2.imread(request.session['custom_image']), (224, 224))
        else:
            start_image = cv2.imread('./DeepFeaturesApp/static/DeepFeaturesApp/pineapple.jpg')
        # creator = ImageFeatureCreator()
        # feature_vector = creator.get_feature_vector(start_image, form.cleaned_data['layer_index'])
        # image = creator.create_from_features(feature_vector, form.cleaned_data['layer_index'],
        #                                      form.cleaned_data['learning_rate'], form.cleaned_data['grad_std_clip'],
        #                                      form.cleaned_data['image_std_clip'], form.cleaned_data['epoch_count'])

        # cv2.imwrite('./DeepFeaturesApp/static/DeepFeaturesApp/image.png', image)
        filename = feature_creations.generate_image_name()
        params = feature_creations.ImageParameters(start_image, filename, form.cleaned_data['learning_rate'],
                                                   form.cleaned_data['layer_index'], form.cleaned_data['image_std_clip'],
                                                   form.cleaned_data['grad_std_clip'], form.cleaned_data['epoch_count'],
                                                   form.cleaned_data['total_variation'])
        global worker_thread_created
        if not worker_thread_created:
            worker_thread_created = True

            worker = feature_creations.AsyncImageFeatureCreator()
            worker.start()

        feature_creations.images_to_make.put(params)

        image_name = os.path.basename(filename)
        return render(request, 'DeepFeaturesApp/index.html', {'form': new_form, 'image_path': image_name})


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
